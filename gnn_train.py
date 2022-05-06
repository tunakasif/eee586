# %%
from torch_geometric.data import Data
import torch
from torch import tensor
import numpy as np

#%matplotlib inline
# import matplotlib.pyplot as plt
from eee586.word_embedding import get_token_encodings
from eee586.utils.adjacency import generate_adj_matrix
from gnn_models import GCN
from gnn_train_utils import train_model, test_model, get_edge_values

# %%
def get_graph_data(
    train_encods: dict,
    test_encods: dict,
    n_train: int = None,
    n_test: int = None,
    window_size=20,
    stride=1,
) -> Data:

    train_docs, test_docs = train_encods.get("input_ids"), test_encods.get("input_ids")

    train_labels, test_labels = train_encods.get("labels"), test_encods.get("labels")

    if n_test is None:
        n_test = len(test_docs)

    if n_train is None:
        n_train = len(train_docs)

    documents = train_docs[:n_train] + test_docs[:n_test]
    labels = train_labels[:n_train] + test_labels[:n_test]

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

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    data.train_mask = tensor(n_train * [True] + (n_nodes - n_train) * [False])
    data.test_mask = tensor(
        n_train * [False] + n_test * [True] + (n_nodes - (n_train + n_test)) * [False]
    )
    return data, all_vocab


#%%
def train_strategy(train_encods, test_encods, n_train=None, n_test=None, together=True):
    """
    If together is True, we will construct single graph for test and train and mask test during training
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if together:
        data, all_vocab = get_graph_data(
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
        loss = train_model(data_train, model, optimizer, criterion)
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
    return model, all_vocab


# %%
train_encods = get_token_encodings("train")
test_encods = get_token_encodings("test")
# %%
model, all_vocab = train_strategy(
    train_encods,
    test_encods,
    n_train=100,
    n_test=20,
    together=True,
)

# %%
def get_gnn_embeddings(model: GCN, train_indices):
    param_list = []
    for param in model.parameters():
        param_list.append(param)
    W1 = param_list[0]
    W2 = param_list[2]
    return W1, W2
