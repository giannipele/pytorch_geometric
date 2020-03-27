import os
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import SplineConv, SAGELafConv, SAGEConv
import math
import numpy as np
import time
from sklearn.metrics import classification_report, f1_score, accuracy_score
from torch.nn import Linear


class GraphSAGE(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden):
        super(GraphSAGE, self).__init__()
        #self.conv1 = SAGEConv(dataset.num_features, hidden)
        self.conv1 = SAGELafConv(dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 2):
            #self.convs.append(SAGEConv(hidden, hidden))
            self.convs.append(SAGELafConv(hidden, hidden))
        self.convn = SAGELafConv(hidden, dataset.num_classes)
        #self.convn = SAGEConv(hidden, dataset.num_classes)
        #self.lin1 = Linear(hidden, hidden)
        #self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.convn.reset_parameters()
        #self.lin1.reset_parameters()
        #self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = F.relu(self.convn(x, edge_index))
        #x = F.relu(self.lin1(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        #x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


EPOCH = 1000
FOLDS = 5
FOLDS_SEED = 196


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
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model(data)[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


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
    for _, mask in data('test_mask'):
        pred = logits[mask].max(1)[1]
        true = data.y[mask]
        y_pred = pred.detach().cpu().clone().numpy()
        y_true = true.detach().cpu().clone().numpy()
        n_classes = np.unique(y_true).size
        target_names = ['class_{}'.format(i) for i in range(n_classes)]
        cr = classification_report(y_true,y_pred, target_names=target_names)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='micro')
        accs.append((cr, acc, f1))
    return accs[0]


def exp(exp_name, seed, style, shared):
    torch.manual_seed(seed)
    res_dir = "results/"

    dataset = 'Cora'
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', dataset)
    dataset = Planetoid(path, dataset, T.NormalizeFeatures())
    data = dataset[0]

    fold = 0
    fold_accuracies = []
    fold_f1 = []
    itr_time = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

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

            model = GraphSAGE(dataset, num_layers=2, hidden=64).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0001)
            best_acc = 0
            count = 0
            for epoch in range(1, EPOCH):
                start_itr = time.time()
                train(model, data, optimizer)
                train_accs = validate(model, data)
                log = 'Epoch: {:03d}, Train: {:.5f}, Validation: {:.5f}'
                print(log.format(epoch, *train_accs))
                log+='\n'
                flog.write(log.format(epoch, *train_accs))
                if train_accs[1] > best_acc:
                    best_acc = train_accs[1]
                    torch.save(model.state_dict(), res_dir + "{}.dth".format(exp_name))
                    print("Saving model at iteration {}".format(epoch))
                    count = 0
                else:
                    count += 1
                if count == 100:
                    break

                itr_time.append(time.time()-start_itr)

            model.load_state_dict(torch.load(res_dir + "{}.dth".format(exp_name)))
            cr, test_acc, test_f1 = test(model, data)
            print(cr)
            flog.write(cr+"\n")
            fold_accuracies.append(test_acc)
            fold_f1.append(test_f1)
        flog.write("----------\n")
        flog.write("Avg Test Accuracy: {:.5f}\tVariance: {:.5f}\n".format(np.mean(fold_accuracies), np.var(fold_accuracies)))
        print("Avg Test Accuracy: {:.5f}\tVariance: {:.5f}\n".format(np.mean(fold_accuracies), np.var(fold_accuracies)))
        flog.write("Avg Test Micro-F1: {:.5f}\tVariance: {:.5f}\n".format(np.mean(fold_f1), np.var(fold_f1)))
        print("Avg Test Micro-F1: {:.5f}\tVariance: {:.5f}\n".format(np.mean(fold_f1), np.var(fold_f1)))

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
    exps = [{'name': 'laf_sage_cora2', "seed": 2603, "style":'frac', "shared":True},
             ]
    main(exps)


