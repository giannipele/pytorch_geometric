import os.path as osp
import time
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Reddit
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import SAGEConv, GATConv, SAGELafConv
from torch.nn import Linear
import numpy as np
import math


class GraphSAGE(torch.nn.Module):
    def __init__(self, seed, in_channels, out_channels):
        super(GraphSAGE, self).__init__()
        #self.conv1 = SAGEConv(in_channels, 16, normalize=False)
        #self.conv2 = SAGEConv(16, out_channels, normalize=False)
        self.conv1 = SAGELafConv(in_channels, 64, normalize=False)
        self.conv2 = SAGELafConv(64, out_channels, normalize=False)
        #self.lin = Linear(64, out_channels)

    def forward(self, x, data_flow):
        data = data_flow[0]
        x = x[data.n_id]
        x = F.relu(self.conv1((x, None), data.edge_index, size=data.size))
        x = F.dropout(x, p=0.5, training=self.training)
        data = data_flow[1]
        x = F.relu(self.conv2((x, None), data.edge_index, size=data.size))
        #x = self.lin(x)
        return F.log_softmax(x, dim=1)


EPOCH = 200
FOLDS = 5
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
    res_dir = "results/"
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit')
    dataset = Reddit(path)
    data = dataset[0]
    loader = NeighborSampler(data, size=[25, 10], num_hops=2, batch_size=1000,
                             shuffle=True, add_self_loops=True)
    fold = 0
    fold_accuracies = []
    itr_time = []

    feats = dataset.num_features
    classes = dataset.num_classes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(res_dir + '{}.log'.format(exp_name), 'w') as flog:
        start_time = time.time()
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

            model = GraphSAGE(seed * fold, feats, classes).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00001)
            best_acc = 0
            count = 0

            for epoch in range(1, EPOCH):
                start_itr = time.time()
                train_acc = train(data, model, loader, optimizer, device)
                val_acc = validate(data, model, loader, device, data.val_mask)
                log = 'Epoch: {:03d}, Train: {:.5f}, Validation: {:.5f}'
                print(log.format(epoch, train_acc, val_acc))
                log+='\n'
                flog.write(log.format(epoch, train_acc, val_acc))
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), res_dir + "{}.dth".format(exp_name))
                    print("Saving model at iteration {}".format(epoch))
                    count = 0
                else:
                    count += 1
                if count == 100:
                    break
                itr_time.append(time.time() - start_itr)

            model.load_state_dict(torch.load(res_dir + "{}.dth".format(exp_name)))
            test_acc = test(data, model, loader, device, data.test_mask)
            print('Test Loss: {:.5f}'.format(test_acc))
            flog.write('Test Loss: {:.5f}\n'.format(test_acc))
            fold_accuracies.append(test_acc)
        flog.write("----------\n")
        flog.write("Avg Test Loss: {:.5f}\tVariance: {:.5f}\n".format(np.mean(fold_accuracies), np.var(fold_accuracies)))
        print("Avg Test Loss: {:.5f}\tVariance: {:.5f}\n".format(np.mean(fold_accuracies), np.var(fold_accuracies)))

        stop_time = time.time() - start_time
        avg_itr_time = np.mean(itr_time)

        flog.write("Avg iteration time: {:.3f} s\n".format(avg_itr_time))
        flog.write("Execution time: {:.3f} s\n".format(stop_time))
        print("Avg iteration time: {:.3f} s\n".format(avg_itr_time))
        print("Execution time: {:.3f} s\n".format(stop_time))


def main(exps):
    for e in exps:
        exp(e['name'], e['seed'], e['style'], e['shared'])


if __name__ == '__main__':
    exps = [{'name': 'laf_sage_reddit', "seed": 2603, "style":'frac', "shared":True},
            ]
    main(exps)



