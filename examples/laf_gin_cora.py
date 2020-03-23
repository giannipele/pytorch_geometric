import os
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import math
import numpy as np
from torch import autograd
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINLafConv, GINConv


class GIN(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, seed):
        super(GIN, self).__init__()
        self.conv1 = GINLafConv(Sequential(
            Linear(dataset.num_features, hidden),
            ReLU(),
            Linear(hidden, hidden),
            ReLU(),
            BN(hidden),
        ),
            train_eps=False, seed=seed)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINLafConv(Sequential(
                    Linear(hidden, hidden),
                    ReLU(),
                    Linear(hidden, hidden),
                    ReLU(),
                    BN(hidden),
                ),
                    train_eps=False))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class GINNet(torch.nn.Module):
    def __init__(self, dataset):
        super(GINNet, self).__init__()
        dim = 16

        nn1 = Sequential(Linear(dataset.num_features, dim), ReLU(), Linear(dim, dim))
        #self.conv1 = GINConv(nn1)
        self.conv1 = GINLafConv(nn1, embed_dim=dataset.num_features)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        #self.conv2 = GINConv(nn2)
        self.conv2 = GINLafConv(nn2, embed_dim=dim)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        #nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        ##self.conv3 = GINConv(nn3)
        #self.conv3 = GINLafConv(nn3, embed_dim=dim)
        #self.bn3 = torch.nn.BatchNorm1d(dim)

        #nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        ##self.conv4 = GINConv(nn4)
        #self.conv4 = GINLafConv(nn4, embed_dim=dim)
        #self.bn4 = torch.nn.BatchNorm1d(dim)

        #nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        ##self.conv5 = GINConv(nn5)
        #self.conv5 = GINLafConv(nn5, embed_dim=dim)
        #self.bn5 = torch.nn.BatchNorm1d(dim)

        #self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, dataset.num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        #x = F.relu(self.conv3(x, edge_index))
        #x = self.bn3(x)
        #x = F.relu(self.conv4(x, edge_index))
        #x = self.bn4(x)
        #x = F.relu(self.conv5(x, edge_index))
        #x = self.bn5(x)
        #x = global_add_pool(x, batch)
        #x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

EPOCH = 200
FOLDS = 5
FOLDS_SEED = 92

def gen_folds(n_data, folds, seed):
    idx = np.random.RandomState(seed=seed).permutation(n_data)
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

        test_idx = test_idx + offset
        val_idx = 0 if val_idx + offset >= n_data else val_idx + offset
        yield (train_mask, val_mask, test_mask)


def train(model, data, optimizer):
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
            #model = GINNet(dataset).to(device)
            model = GIN(dataset, 2, 64, seed).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
            best_acc = 0
            count = 0
            for epoch in range(1, EPOCH):
                train(model, data, optimizer)
                train_accs = validate(model, data)
                log = 'Epoch: {:03d}, Train: {:.4f}, Validation: {:.4f}'
                print(log.format(epoch, *train_accs))
                log += '\n'
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
    exps = [{'name': 'laf_gin_cora_2303', "seed": 2303, "style":'frac', "shared":True},
            ]
    main(exps)


