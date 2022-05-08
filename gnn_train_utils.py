import torch
from torch import tensor
from gnn_models import GCN


def train_model(data, model, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model.forward()
    loss = criterion(out[data.train_idx], data.y[data.train_idx])
    loss.backward()
    optimizer.step()
    return loss


def train_model_mlp(model, data, optimizer, criterion, labels):
    model.train()
    optimizer.zero_grad()
    out = model.forward(data)
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()
    return loss


def test_model_mlp(model, data, train_labels=None, test_labels=None, type=None):
    model.eval()
    out = model.forward(data)
    pred = out.argmax(dim=1)
    if type == "test":
        correct = pred == test_labels
        acc = int(correct.sum()) / len(test_labels)
    else:
        correct = pred == train_labels
        acc = int(correct.sum()) / len(train_labels)
    return acc * 100


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


def get_gnn_embeddings(model: GCN, n_train, n_test):
    param_list = []
    for param in model.parameters():
        param_list.append(param)
    W1 = param_list[1]
    gnn_embed_train = W1[:, :n_train]
    gnn_embed_test = W1[:, n_train : (n_train + n_test)]
    return gnn_embed_train.T, gnn_embed_test.T
