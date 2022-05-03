# %%
import numpy as np
import torch
from scipy.sparse import csr_matrix
import itertools
from torch_geometric.data import Data


from eee586.word_embedding import get_token_encodings

train_encodings = get_token_encodings("train")
test_encodings = get_token_encodings("train")
# %%
def get_adjacency_matrix():
    pass


def get_graph_data(
    train_encodings: dict,
    test_encodings: dict,
    num_of_train_samples: int = None,
    num_of_test_samples: int = None,
):
    """
    Return the Data class of the pytorch geometric
    """
    train_documents, test_documents = train_encodings.get(
        "input_ids"
    ), test_encodings.get("input_ids")
    train_labels, test_labels = train_encodings.get("labels"), test_encodings.get(
        "labels"
    )

    if num_of_train_samples and num_of_test_samples is None:
        num_of_train_samples = len(train_documents)
        num_of_test_samples = len(documents)

    documents = train_documents + test_documents
    labels = train_labels, test_labels

    vocab_list = [set(document) for document in documents]
    vocab = list(set().union(*vocab_list))

    A = get_adjacency_matrix(documents)
    smat = csr_matrix(A)
    values, column_indices, row_dptrs = smat.data, smat.indices, smat.indptr

    t = []
    for i in range(0, len(row_dptrs) - 1):
        repetition = row_dptrs[i + 1] - row_dptrs[i]
        for j in range(repetition):
            t.append(i)

    t = torch.tensor(t).reshape(-1, 1)
    column_indices = torch.tensor(column_indices).reshape(-1, 1)

    edge_index = torch.concat((t, column_indices), dim=1).T
    edge_attr = torch.tensor(values).reshape(-1, 1)

    num_of_nodes = A.shape[0]
    x = torch.eye(num_of_nodes)
    y = torch.tensor(labels)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.train_idx = torch.arange(len(train_labels))
    data.test_idx = torch.arange(
        len(train_labels), len(train_labels) + len(test_labels)
    )
    return data


def find_pmi_score(
    w1_id: int,
    w2_id: int,
    documents: list[list[int]],
    window_lenght: int = 10,
    stride: int = 1,
):
    corpus = list(itertools.chain(*documents))
    p_i, p_j, p_ij = 0, 0, 0
    total_num_of_windows = (len(corpus) - window_lenght + 1) / stride

    if w1_id == w2_id:
        pmi_score = 1
    else:
        for i in range(0, len(corpus) - window_lenght, stride):
            if w1_id in corpus[i : i + window_lenght]:
                p_i += 1
            if w2_id in corpus[i : i + window_lenght]:
                p_j += 1
            if (
                w1_id in corpus[i : i + window_lenght]
                and w2_id in corpus[i : i + window_lenght]
            ):
                p_ij += 1
        pmi_score = np.log(p_ij / (p_i * p_j * total_num_of_windows))
    return pmi_score


def find_tf_idf_score(
    documents: list[list[int]],
    document_index: int,
    word_id: int,
):
    """
    t — term (word)
    d — document (set of words)
    N — count of corpus
    corpus — the total document set
    """
    corpus = np.array(list(itertools.chain(*documents)))

    d = np.array(documents[document_index])
    tf = np.count_nonzero(d == word_id) / len(d)
    df = np.count_nonzero(corpus == word_id) / len(d)
    idf = np.log(len(documents) / (df + 1))
    tf_idf_score = tf * idf
    return tf_idf_score


# # %%

# # %%
# from torch_geometric.datasets import Planetoid

# dataset = Planetoid(root="/tmp/Cora", name="Cora")
# # %%
# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv


# class GCN(torch.nn.Module):
#     def __init__(self, edge_weight):
#         super().__init__()
#         self.conv1 = GCNConv(dataset.num_node_features, 16)
#         self.conv2 = GCNConv(16, dataset.num_classes)
#         self.edge_weight = edge_weight

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index

#         x = self.conv1(x, edge_index, self.edge_weight)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index, self.edge_weight)

#         return F.log_softmax(x, dim=1)


# # %%
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# document = None  # get from tuna
# edge_weight, edge_index = get_csr_matrix(document)
# model = GCN(edge_weight).to(device)
# data = dataset[0].to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# model.train()
# for epoch in range(200):
#     optimizer.zero_grad()
#     out = model(data)
#     loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
#     loss.backward()
#     optimizer.step()
# # %%
# model.eval()
# pred = model(data).argmax(dim=1)
# correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
# acc = int(correct) / int(data.test_mask.sum())
# print(f"Accuracy: {acc:.4f}")
# # %%

# # %%

# # %%

# # %%
