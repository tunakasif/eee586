# %%
import numpy as np
import torch
from scipy.sparse import csr_matrix

mat = np.random.randint(low=0, high=3, size=(5, 5))
smat = csr_matrix(mat)
# %%
def get_csr_matrix(document: torch.Tensor):
    """
    Generate COO format for adjacency matrix to pass outputs to the Data Class in Pytorch Geometric
    """
    # find the first zero element and clip
    document = document.numpy()
    document = document[: np.argmax(document == 0)]
    vocab = list(set(document))
    N = len(vocab)
    A = np.eye(N)

    ## find i,j entry of the document by sliding window
    for i in range(N):
        for j in range(N):
            A[i, j] = find_pmi(vocab[i], vocab[j], document)
    smat = csr_matrix(A)
    values, column_indices, row_dptrs = smat.data, smat.indices, smat.indptr

    t = []
    for i in range(0, len(row_dptrs) - 1):
        repetition = row_dptrs[i + 1] - row_dptrs[i]
        for j in range(repetition):
            t.append(i)
    t = torch.tensor(t).reshape(-1, 1)
    values = torch.tensor(values).reshape(-1, 1)
    column_indices = torch.tensor(column_indices).reshape(-1, 1)

    edge_index = torch.concat((t, column_indices), dim=1).T
    return values, edge_index


def find_pmi(w1_id, w2_id, document, window_lenght=10):
    p_i, p_j, p_ij = 0, 0, 0
    total_num_of_windows = len(document) - 10
    if w1_id == w2_id:
        pmi_score = 1
    else:
        for i in range(0, len(document) - window_lenght):
            if w1_id == document[i]:
                p_i += 1
            if w2_id == document[i]:
                p_j += 1
            if (
                w1_id in document[i:, +window_lenght]
                and w1_id in document[i:, +window_lenght]
            ):
                p_ij += 1
        pmi_score = np.log(p_ij / (p_i * p_j * total_num_of_windows))
    return pmi_score


# %%
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root="/tmp/Cora", name="Cora")
# %%
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, edge_weight):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)
        self.edge_weight = edge_weight

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index, self.edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, self.edge_weight)

        return F.log_softmax(x, dim=1)


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
document = None  # get from tuna
edge_weight, edge_index = get_csr_matrix(document)
model = GCN(edge_weight).to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
# %%
model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f"Accuracy: {acc:.4f}")
# %%

# %%

# %%

# %%
