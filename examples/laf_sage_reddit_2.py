import os.path as osp

import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Reddit
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import SAGEConv, GATConv, SAGELafConv
import os
import numpy as np
import math

#parser = argparse.ArgumentParser()
#parser.add_argument('--model', type=str, default='SAGE')
#args = parser.parse_args()



class SAGENet(torch.nn.Module):
    def __init__(self, seed, in_channels, out_channels):
        super(SAGENet, self).__init__()
        #self.conv1 = SAGEConv(in_channels, 16, normalize=False)
        #self.conv2 = SAGEConv(16, out_channels, normalize=False)
        self.conv1 = SAGELafConv(in_channels, 16, normalize=False)
        self.conv2 = SAGELafConv(16, out_channels, normalize=False)

    def forward(self, x, data_flow):
        data = data_flow[0]
        x = x[data.n_id]
        x = F.relu(self.conv1((x, None), data.edge_index, size=data.size))
        x = F.dropout(x, p=0.5, training=self.training)
        data = data_flow[1]
        x = self.conv2((x, None), data.edge_index, size=data.size)
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



def train(data, model, loader, optimizer, device):
    model.train()

    total_loss = 0
    for data_flow in loader(data.train_mask):
        optimizer.zero_grad()
        out = model(data.x.to(device), data_flow.to(device))
        loss = F.nll_loss(out, data.y[data_flow.n_id].to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data_flow.batch_size
    return total_loss / data.train_mask.sum().item()


def test(data, model, loader, device, mask):
    model.eval()

    correct = 0
    for data_flow in loader(mask):
        pred = model(data.x.to(device), data_flow.to(device)).max(1)[1]
        correct += pred.eq(data.y[data_flow.n_id].to(device)).sum().item()
    return correct / mask.sum().item()


def validate(data, model, loader, device, mask):
    model.eval()

    correct = 0
    for data_flow in loader(mask):
        pred = model(data.x.to(device), data_flow.to(device)).max(1)[1]
        correct += pred.eq(data.y[data_flow.n_id].to(device)).sum().item()
    return correct / mask.sum().item()


def exp(exp_name, seed, style, shared):
    torch.manual_seed(seed)
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit')
    dataset = Reddit(path)
    data = dataset[0]
    loader = NeighborSampler(data, size=[25, 10], num_hops=2, batch_size=1000,
                             shuffle=True, add_self_loops=True)
    data = dataset[0]
    fold = 0
    accuracies = []
    feats = dataset.num_features
    classes = dataset.num_classes
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

            #data = data.to(device)
            #model = SAGENet(dataset, seed*fold, style, shared).to(device)
            model = SAGENet(seed*fold, feats, classes).to(device)
            #print(list(model.parameters()))
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00001)
            best_acc = 0
            count = 0
            for epoch in range(1, EPOCH):
                #print(list(model.conv1.aggregation.parameters()))
                train_acc = train(data, model, loader, optimizer, device)
                val_acc = validate(data, model, loader, device, data.val_mask)
                log = 'Epoch: {:03d}, Train: {:.4f}, Validation: {:.4f}'
                print(log.format(epoch, train_acc, val_acc))
                log+='\n'
                flog.write(log.format(epoch, train_acc, val_acc))
                if val_acc < best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), "{}.dth".format(exp_name))
                    print("Saving model at iteration {}".format(epoch))
                    count = 0
                else:
                    count += 1
                if count == 200:
                    break

            model.load_state_dict(torch.load("{}.dth".format(exp_name)))
            test_acc = test(data, model, loader, device, data.test_mask)
            print('Test Loss: {}'.format(test_acc))
            flog.write('Test Loss: {}\n'.format(test_acc))
            accuracies.append(test_acc)
        flog.write("----------\n")
        flog.write("Avg Test Loss: {}\tVariance: {}\n".format(np.mean(accuracies), np.var(accuracies)))


def main(exps):
    for e in exps:
        exp(e['name'], e['seed'], e['style'], e['shared'])


if __name__ == '__main__':
    exps = [{'name': 'sage_cora_2403', "seed": 2403, "style":'frac', "shared":True},
            ]
    main(exps)



