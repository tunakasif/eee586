# %%
from scipy.sparse import csr_matrix
from torch_geometric.data import Data
from pathlib import Path
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from scipy.sparse import random
from torch import tensor
from torch_sparse import SparseTensor
import numpy as np

#%matplotlib inline
# import matplotlib.pyplot as plt
from eee586 import PKL_DIR, WORK_DIR
from eee586.word_embedding import get_token_encodings
from eee586.utils.adjacency import generate_adj_matrix
from eee586.utils.generic import pickle_dump, pickle_load

# %%
def train_func(data, model, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model.forward()
    loss = criterion(out[data.train_idx], data.y[data.train_idx])
    loss.backward()
    optimizer.step()
    return loss


def test_model(data, model, type):
    model.eval()
    out = model()
    pred = out.argmax(dim=1)
    if type == "test":
        correct = pred[data.test_idx] == data.y[data.test_idx]
        acc = int(correct.sum()) / int(data.test_idx.sum())
    else:
        correct = pred[data.train_idx] == data.y[data.train_idx]
        acc = int(correct.sum()) / int(data.train_idx.sum())
    return pred, acc * 100


def get_edge_values(c):
    row, col = tensor(c.row).reshape(-1, 1), tensor(c.col).reshape(-1, 1)
    data = c.data
    edge_index = torch.concat((row, col), dim=1).T.long().contiguous()
    edge_attr = tensor(data).reshape(-1)
    return edge_index, edge_attr


def get_graph_data(
    train_encods: dict,
    test_encods: dict,
    n_train: int = None,
    n_test: int = None,
    window_size=10,
    stride=1,
) -> Data:

    train_docs, test_docs = np.array(train_encods.get("input_ids")), np.array(
        test_encods.get("input_ids")
    )
    train_labels, test_labels = np.array(train_encods.get("labels")), np.array(
        test_encods.get("labels")
    )

    if n_test is None:
        n_test = len(test_docs)

    if n_train is None:
        n_train = (len(train_docs),)

    train_indices = np.random.choice(len(train_docs), n_train, replace=False)
    test_indices = np.random.choice(len(test_docs), n_test, replace=False)
    documents = list(
        np.concatenate((train_docs[train_indices], test_docs[test_indices]), axis=0)
    )
    labels = list(
        np.concatenate((train_labels[train_indices], test_labels[test_indices]), axis=0)
    )

    doc_vocabs = [set(doc) for doc in documents]
    all_vocab = list(set.union(*doc_vocabs))
    n_nodes = len(all_vocab) + len(documents)

    c = generate_adj_matrix(
        documents,
        dataset_name=f"SetFit/20_newsgroups_ntrain{n_train}_ntest{n_test}",
        window_size=window_size,
        stride=stride,
    )
    edge_index, edge_attr = get_edge_values(c)
    del c
    x = torch.eye(n_nodes)
    y = tensor(labels + (n_nodes - len(labels)) * [0])
    print(x.shape, y.shape)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    data.train_idx = tensor(n_train * [True] + (n_nodes - n_train) * [False])
    data.test_idx = tensor(
        n_train * [False] + n_test * [True] + (n_nodes - (n_train + n_test)) * [False]
    )
    return data, train_indices, test_indices


# def get_graph_data_train(
#     train_encods: dict,
#     n_train: int = None,
#     ratio=0.8,
#     window_size=10,
#     stride=1,
# ) -> Data:

#     """
#     n_train: used to select subset of train dataset to fit the memory
#     ratio: used to divide train set to train and validation
#     """
#     train_docs, train_labels = train_encods.get("input_ids"), train_encods.get("labels")
#     if n_train is None:
#         n_train = len(train_docs)
#     train_docs, train_labels = train_docs[:n_train], train_labels[:n_train]

#     idx = int(len(train_docs) * (ratio))

#     documents, labels = train_docs, train_labels

#     train_docs, val_docs = train_docs[:idx], train_docs[idx:]
#     train_labels, val_labels = train_labels[:idx], train_labels[idx:]
#     n_val, n_train = len(val_docs), len(train_docs)

#     A = generate_adj_matrix(
#         documents,
#         dataset_name=f"SetFit/20_newsgroups_ntrain{n_train}_tv_ratio{ratio}",
#         window_size=window_size,
#         stride=stride,
#     )
#     edge_index, edge_attr, n_nodes = get_edge_values(A)
#     x, y = torch.eye(n_nodes,300), tensor(labels + (n_nodes - len(labels)) * [0])

#     data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

#     data.train_idx = tensor(n_train * [True] + (n_nodes - n_train) * [False])
#     data.test_idx = tensor(
#         n_train * [False] + n_val * [True] + (n_nodes - (n_train + n_val)) * [False]
#     )
#     return data


# def get_graph_data_test(
#     test_encods: dict,
#     n_test: int = None,
#     window_size=10,
#     stride=1,
# ) -> Data:
#     test_docs, test_labels = test_encods.get("input_ids"), test_encods.get("labels")
#     if n_test is None:
#         n_test = len(test_docs)
#     documents, labels = test_docs[:n_test], test_labels[:n_test]

#     A = generate_adj_matrix(
#         documents,
#         dataset_name=f"SetFit/20_newsgroups_ntest{n_test}",
#         window_size=window_size,
#         stride=stride,
#     )

#     edge_index, edge_attr, n_nodes = get_edge_values(A)

#     x, y = torch.eye(n_nodes), tensor(labels + (n_nodes - len(labels)) * [0])
#     data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
#     data.test_idx = tensor(n_test * [True] + (n_nodes - n_test) * [False])
#     data.train_idx = tensor(n_nodes * [False])

#     return data


# %%
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, data: Data):
        super().__init__()
        self.data = data
        # self.edge_weight = torch.nn.Parameter(self.data.edge_attr)
        self.edge_weight = self.data.edge_attr
        self.conv1 = GCNConv(data.num_node_features, hidden_channels, cached=True)
        self.conv2 = GCNConv(hidden_channels, 20, cached=True)

    def forward(self):
        x, edge_index = (self.data.x, self.data.edge_index)
        x = x.double()
        x = F.relu(self.conv1(x, edge_index, self.edge_weight))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index, self.edge_weight)
        x = F.dropout(x, p=0.6, training=self.training)
        return x


class GAT(torch.nn.Module):
    def __init__(self, nhid, data: Data, dropout=True):
        super(GAT, self).__init__()
        self.data = data
        self.conv1 = GATConv(data.num_node_features, nhid, heads=1, dropout=dropout)
        self.conv2 = GATConv(nhid * 1, 20, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self):
        x, edge_index, edge_attr = (
            self.data.x,
            self.data.edge_index,
            self.data.edge_attr,
        )
        x = x.double()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


#%%
def train_strategy(train_encods, test_encods, n_train=None, n_test=None, together=True):
    """
    If together is True, we will construct single graph for test and train and mask test during training
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if together:
        data, train_indices, test_indices = get_graph_data(
            train_encods,
            test_encods,
            n_train=n_train,
            n_test=n_test,
            window_size=20,
            stride=1,
        )
        data_train = data.to(device)

    # else:
    #     data_train = get_graph_data_train(
    #         train_encods, n_train=100, ratio=0.8, window_size=10, stride=1
    #     ).to(device)
    #     data_test = get_graph_data_test(
    #         test_encods, n_test=40, window_size=10, stride=1
    #     ).to(device)

    model = GCN(data=data_train, hidden_channels=200).double().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(0, 50):
        loss = train_func(data_train, model, optimizer, criterion)
        if together == True:
            _, train_acc = test_model(data_train, model, type="train")
            _, test_acc = test_model(data_train, model, type="test")
            if epoch % 10 == 0:
                print(f"Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")
                print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")

        # else:
        #     _, train_acc = test_model(data_train, model, type="train")
        #     _, test_acc = test_model(data_test, model, type="test")

        #     print(f"Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")
        #     if epoch % 10 == 0:
        #         print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")
    return model


# %%
train_encods = get_token_encodings("train")
test_encods = get_token_encodings("test")
# %%
model = train_strategy(
    train_encods,
    test_encods,
    n_train=100,
    n_test=30,
    together=True,
)

# %%
