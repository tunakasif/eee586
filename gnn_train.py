# %%
from torch_geometric.data import Data
import torch
from torch import tensor
import numpy as np

#%matplotlib inline
# import matplotlib.pyplot as plt
from eee586.word_embedding import (
    get_token_encodings,
    get_doc_embeddings,
)
from eee586.utils.adjacency import generate_adj_matrix
from gnn_models import GCN, MLP
from gnn_train_utils import (
    train_model,
    test_model,
    get_edge_values,
    train_model_mlp,
    test_model_mlp,
    get_gnn_embeddings,
)

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

    data.train_idx = tensor(n_train * [True] + (n_nodes - n_train) * [False])
    data.test_idx = tensor(
        n_train * [False] + n_test * [True] + (n_nodes - (n_train + n_test)) * [False]
    )
    return data, all_vocab


#%%
def train_strategy(
    train_encods, test_encods, hidden_channels, n_train=None, n_test=None, together=True
):
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

    model = GCN(layer_no=2, data=data_train).double().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    for epoch in range(0, 10):
        loss = train_model(data_train, model, optimizer, criterion)
        if together == True:
            _, train_acc = test_model(data_train, model, type="train")
            _, test_acc = test_model(data_train, model, type="test")
            if epoch % 5 == 0:
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

n_train = 500
n_test = 50
model, all_vocab = train_strategy(
    train_encods,
    test_encods,
    hidden_channels=[2000, 200, 20],
    n_train=n_train,
    n_test=n_test,
    together=True,
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
bert_embed_full_train, bert_embed_full_test = get_doc_embeddings(
    "train"
), get_doc_embeddings("test")
bert_embed_train = tensor(bert_embed_full_train[:n_train]).to(device)
bert_embed_test = tensor(bert_embed_full_test[:n_test]).to(device)
gnn_embed_train, gnn_embed_test = get_gnn_embeddings(model, n_train, n_test)

embed_out_train = torch.concat((bert_embed_train, gnn_embed_train), dim=1)
embed_out_test = torch.concat((bert_embed_test, gnn_embed_test), dim=1)
train_labels = tensor(train_encods.get("labels")[:n_train]).to(device)
test_labels = tensor(test_encods.get("labels")[:n_test]).to(device)

#%%
# Now pass this output to MLP and train
model_mlp = MLP(input_dim=embed_out_train.shape[1])
optimizer = torch.optim.Adam(model_mlp.parameters(), lr=0.01, weight_decay=0)
criterion = torch.nn.CrossEntropyLoss()
model_mlp = model_mlp.to(device)
criterion = criterion.to(device)

for epoch in range(0, 1000):
    loss = train_model_mlp(
        model_mlp, embed_out_train, optimizer, criterion, train_labels
    )
    train_acc = test_model_mlp(
        model_mlp, embed_out_train, train_labels=train_labels, type="train"
    )
    test_acc = test_model_mlp(
        model_mlp, embed_out_test, test_labels=test_labels, type="test"
    )
    if epoch % 100 == 0:
        print(f"Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")

# %%
