import torch
from torch import tensor


def train_model(data, model, optimizer, criterion):
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
