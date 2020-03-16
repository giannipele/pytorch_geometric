import os
import torch
torch.version.cuda=None
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import SplineConv, SAGELafConv, SAGEConv
import math
import numpy as np
import sys
import pdb
import traceback
from torch import autograd

class SAGENet(torch.nn.Module):
    def __init__(self, dataset, seed, style, shared):
        super(SAGENet, self).__init__()
        self.conv1 = SAGELafConv(dataset.num_features, 16, seed=seed, style=style, shared=shared)
        self.conv2 = SAGELafConv(16, dataset.num_classes, seed=seed, style=style, shared=shared)
        #self.conv1 = SAGEConv(dataset.num_features, 16)
        #self.conv2 = SAGEConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

EPOCH = 200
FOLDS = 10
FOLDS_SEED = 92

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

def train(model, data, optimizer):
    with autograd.detect_anomaly():
        model.train()
        optimizer.zero_grad()
        F.nll_loss(model(data)[data.train_mask], data.y[data.train_mask]).backward()
        optimizer.step()

    #par = []
    #for p in model.conv1.aggregation.parameters():
    #    par.append(p)
    #print(par)

def validate(model, data):
    model.eval()
    logits, accs = model(data), []
    for _, mask in data('train_mask', 'val_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

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
    dataset = 'Cora'
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', dataset)
    dataset = Planetoid(path, dataset, T.NormalizeFeatures())
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

            data = data.to(device)
            model = SAGENet(dataset, seed*fold, style, shared).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
            best_acc = 0
            count = 0
            for epoch in range(1, EPOCH):
                train(model, data, optimizer)
                train_accs = validate(model, data)
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
                if count == 50:
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
    exps = [{'name': 'frac_shared_1603', "seed": 1603, "style":'frac', "shared":True},
             ]
    main(exps)


