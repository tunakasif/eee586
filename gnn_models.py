import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data


class GCN(torch.nn.Module):
    def __init__(self, layer_no, data: Data):
        super().__init__()
        self.data = data
        # self.edge_weight = torch.nn.Parameter(self.data.edge_attr)
        self.edge_weight = self.data.edge_attr
        self.layer_no = layer_no
        if layer_no == 3:
            self.conv1 = GCNConv(data.num_node_features, 2000, cached=True)
            self.conv2 = GCNConv(2000, 200, cached=True)
            self.conv3 = GCNConv(200, 20, cached=True)
        else:
            self.conv1 = GCNConv(data.num_node_features, 200, cached=True)
            self.conv2 = GCNConv(200, 20, cached=True)

    def forward(self):
        x, edge_index = (self.data.x, self.data.edge_index)
        x = x.double()
        x = F.relu(self.conv1(x, edge_index, self.edge_weight))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, self.edge_weight)
        x = F.dropout(x, p=0.5, training=self.training)
        if self.layer_no == 3:
            x = self.conv3(x, edge_index, self.edge_weight)
            x = F.dropout(x, p=0.5, training=self.training)
        return x


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
