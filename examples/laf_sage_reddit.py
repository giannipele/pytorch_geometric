import os
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import SAGELafConv, SAGEConv
from torch_geometric.datasets import Reddit
from torch_geometric.data import NeighborSampler
import math
import numpy as np
import sys
import pdb
import traceback
from torch import autograd
from torch.nn import Linear

class GraphSAGE(torch.nn.Module):
    def __init__(self, dataset, seed, num_layers, hidden):
        super(GraphSAGE, self).__init__()
        #self.conv1 = SAGEConv(dataset.num_features, hidden)
        self.conv1 = SAGELafConv(dataset.num_features, hidden, seed=seed)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            #self.convs.append(SAGEConv(hidden, hidden))
            self.convs.append(SAGELafConv(hidden, hidden))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, data_flow):
        data = data_flow[0]
        x = x[data.n_id]
        edge_index = data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = F.relu(self.lin1(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class SAGENet(torch.nn.Module):
    def __init__(self, dataset, seed, style, shared):
        super(SAGENet, self).__init__()
        #self.conv1 = SAGELafConv(dataset.num_features, 16, seed=seed, style=style, shared=shared)
        #self.conv2 = SAGELafConv(16, dataset.num_classes, seed=seed, style=style, shared=shared)
        self.conv1 = SAGEConv(dataset.num_features, 128)
        self.conv2 = SAGEConv(128, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

EPOCH = 1000
FOLDS = 10
FOLDS_SEED = 196


def gen_folds(n_data, folds, seed):
    idx = np.random.RandomState(seed=seed).permutation(n_data)
    #idx = torch.randperm(n_data)
    # idx = torch.tensor([i for i in range(n_data)])
    offset = math.ceil(n_data / folds)
    test_idx = 0
    val_idx = test_idx + offset
    for i in range(folds):
        val_mask = torch.zeros(n_data, dtype=torch.bool)
        test_mask = torch.zeros(n_data, dtype=torch.bool)

        test_end = min(test_idx + offset, n_data)
        test_mask[idx[test_idx:test_end]] = True

        val_end = min(val_idx + offset, n_data)
        val_mask[idx[val_idx:val_end]] = True

        rest = val_mask + test_mask
        train_mask = ~rest

        # print(val_idx,val_end)
        # print(test_idx, test_end)
        # print("TR:", train_mask)
        # print("VL:", val_mask)
        # print("TS:", test_mask)
        # print("=============================")

        test_idx = test_idx + offset
        val_idx = 0 if val_idx + offset >= n_data else val_idx + offset
        yield (train_mask, val_mask, test_mask)


def train(model, data, optimizer, loader, device):
    total_loss = 0
    for data_flow in loader(data.train_mask):
        optimizer.zero_grad()
        out = model(data.x.to(device), data_flow.to(device))
        loss = F.nll_loss(out, data.y[data_flow.n_id].to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data_flow.batch_size
    return total_loss / data.train_mask.sum().item()


def validate(model, data, loader, device):
    model.eval()
    correct = 0
    mask = data.val_mask
    for data_flow in loader(mask):
        pred = model(data.x.to(device), data_flow.to(device)).max(1)[1]
        correct += pred.eq(data.y[data_flow.n_id].to(device)).sum().item()
    return correct / mask.sum().item()


def test(model, data):
    model.eval()
    logits, accs = model(data), []
    for _, mask in data('train_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


def exp(exp_name, seed, style, shared):
    torch.manual_seed(seed)
    dataset = 'Reddit'
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', dataset)
    dataset = Reddit(path)
    data = dataset[0]
    loader = NeighborSampler(data, size=[25, 10], num_hops=2, batch_size=1000,
                             shuffle=True, add_self_loops=True)
    data = dataset[0]
    fold = 0
    accuracies = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open('{}.log'.format(exp_name), 'w') as flog:
        for tr_mask, vl_mask, ts_mask in gen_folds(data.x.shape[0], FOLDS, FOLDS_SEED):
            fold += 1
            print("FOLD:", fold)
            flog.write("fold #{}\n".format(fold))

            data.train_mask = tr_mask
            data.val_mask = vl_mask
            data.test_mask = ts_mask

            print('Train: {}'.format(torch.sum(data.train_mask)))
            print('Validation: {}'.format(torch.sum(data.val_mask)))
            print('Test: {}'.format(torch.sum(data.test_mask)))
            flog.write('train: {}\n'.format(torch.sum(data.train_mask)))
            flog.write('validation: {}\n'.format(torch.sum(data.val_mask)))
            flog.write('test: {}\n'.format(torch.sum(data.test_mask)))

            data = data.to(device)
            #model = SAGENet(dataset, seed*fold, style, shared).to(device)
            model = GraphSAGE(dataset, seed*fold, num_layers=2, hidden=64).to(device)
            #print(list(model.parameters()))
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
            best_acc = 0
            count = 0
            for epoch in range(1, EPOCH):
                print(list(model.conv1.aggregation.parameters()))
                train(model, data, optimizer, loader, device)
                train_accs = validate(model, data, loader, device)
                log = 'Epoch: {:03d}, Train: {:.4f}, Validation: {:.4f}'
                print(log.format(epoch, *train_accs))
                log+='\n'
                flog.write(log.format(epoch, *train_accs))
                if train_accs[1] > best_acc:
                    best_acc = train_accs[1]
                    torch.save(model.state_dict(), "{}.dth".format(exp_name))
                    print("Saving model at iteration {}".format(epoch))
                    count = 0
                else:
                    count += 1
                if count == 200:
                    break

            model.load_state_dict(torch.load("{}.dth".format(exp_name)))
            accs = test(model, data)
            print('Test Accuracy: {}'.format(accs[1]))
            flog.write('Test Accuracy: {}\n'.format(accs[1]))
            accuracies.append(accs[1])
        flog.write("----------\n")
        flog.write("Avg Test Accuracy: {}\tVariance: {}\n".format(np.mean(accuracies), np.var(accuracies)))


def main(exps):
    for e in exps:
        exp(e['name'], e['seed'], e['style'], e['shared'])


if __name__ == '__main__':
    exps = [{'name': 'sage_cora_2403', "seed": 2403, "style":'frac', "shared":True},
             ]
    main(exps)


