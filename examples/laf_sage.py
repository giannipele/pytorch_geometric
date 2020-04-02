import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Reddit, PPI
import torch_geometric.transforms as T
from torch_geometric.nn import SplineConv, SAGELafConv, SAGEConvCorrect
import math
import numpy as np
import time
from sklearn.metrics import classification_report, f1_score, accuracy_score
from torch_geometric.data import NeighborSampler
from torch.nn import Linear
from torch_geometric.data import InMemoryDataset, Data

EPOCH = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_LAYER = 2


class GraphSAGE(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConvCorrect(dataset.num_features, hidden)
        #self.conv1 = SAGELafConv(dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGEConvCorrect(hidden, hidden))
            #self.convs.append(SAGELafConv(hidden, hidden))
        self.lin = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, data_flow):
        data = data_flow[0]
        x = x[data.n_id]
        #x = F.relu(self.conv1((x, None), data.edge_index, size=data.size))
        x = F.relu(self.conv1((x,None), data.edge_index, size=data.size))
        #x = F.dropout(x, p=0.5, training=self.training)
        i = 1
        for conv in self.convs:
            data = data_flow[i]
            i+=1
            x = F.relu(conv((x,None), data.edge_index, size=data.size))
        x = F.relu(self.lin(x))
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__




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


def get_dataset(name, num_hops):
    dataset = None
    loader = None
    if name.lower() == 'cora':
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
        dataset = Planetoid(path, name, T.NormalizeFeatures())
        train_mask = torch.zeros(dataset.data.x.shape[0], dtype=torch.long)
        val_mask = torch.zeros(dataset.data.x.shape[0], dtype=torch.long)
        test_mask = torch.zeros(dataset.data.x.shape[0], dtype=torch.long)
        train_mask[:1000] = True
        val_mask[1000:1500] = True
        test_mask[1500:2000] = True
        dataset.data.train_mask = train_mask
        dataset.data.val_mask = val_mask
        dataset.data.test_mask = test_mask
        loader = NeighborSampler(dataset[0], size=[25, 10], num_hops=num_hops, batch_size=500,
                                 shuffle=True, add_self_loops=True)
    elif name.lower() == 'reddit':
        #path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'RedditSage')
        dataset = RedditSage(path)
        loader = NeighborSampler(dataset[0], size=[25, 10], num_hops=num_hops, batch_size=1000,
                                 shuffle=True, add_self_loops=True)
    return dataset, loader


def train2(data, model, loader, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    for data_flow in loader(data.train_mask):
        optimizer.zero_grad()
        out = model(data.x.to(DEVICE), data_flow.to(DEVICE))
        loss = F.nll_loss(out, data.y[data_flow.n_id].to(DEVICE))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data_flow.batch_size
        pred = out.max(1)[1]
        correct += pred.eq(data.y[data_flow.n_id].to(DEVICE)).sum().item()
    return total_loss / data.train_mask.sum().item(), correct / data.train_mask.sum().item()


def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model(data)[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


def validate2(data, model, loader):
    model.eval()
    total_loss = 0
    correct = 0
    for data_flow in loader(data.val_mask):
        out = model(data.x.to(DEVICE), data_flow.to(DEVICE))
        loss = F.nll_loss(out, data.y[data_flow.n_id].to(DEVICE))
        total_loss += loss.item() * data_flow.batch_size
        pred = out.max(1)[1]
        correct += pred.eq(data.y[data_flow.n_id].to(DEVICE)).sum().item()
    return total_loss / data.val_mask.sum().item(), correct / data.val_mask.sum().item()


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
        cr = classification_report(y_true, y_pred, target_names=target_names)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='micro')
        accs.append((cr, acc, f1))
    return accs[0]


def test2(data, model, loader):
    model.eval()
    for data_flow in loader(data.test_mask):
        pred = model(data.x.to(DEVICE), data_flow.to(DEVICE)).max(1)[1]
        true = data.y[data_flow.n_id]
        y_pred = pred.detach().cpu().clone().numpy()
        y_true = true.detach().cpu().clone().numpy()
        n_classes = max(np.unique(y_pred).size, np.unique(y_true).size)
        target_names = ['class_{}'.format(i) for i in range(n_classes)]
        cr = classification_report(y_true, y_pred, target_names=target_names)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='micro')
        return cr, acc, f1


def exp(exp_name, dataset_name, seed, style, shared):
    torch.manual_seed(seed)
    res_dir = "results/"
    start_time = time.time()

    dataset, loader = get_dataset(dataset_name, num_hops=NUM_LAYER)
    data = dataset[0]

    itr_time = []

    data = data.to(DEVICE)

    print("Loading {} dataset took {} s".format(dataset_name, time.time()-start_time))

    model = GraphSAGE(dataset, num_layers=NUM_LAYER, hidden=64).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0001)
    best_acc = 0
    count = 0

    with open(res_dir + '{}.log'.format(exp_name), 'w') as flog:
        for epoch in range(1, EPOCH+1):
            start_itr = time.time()
            tr_loss, tr_acc = train2(data, model, loader, optimizer)
            val_loss, val_acc = validate2(data, model, loader)
            log = 'Epoch: {:03d}, Train Loss: {:.5f}, Validation Loss: {:.5f}, Train Acc: {:.5f}, Validation Acc: {:.5f} '
            print(log.format(epoch, tr_loss, val_loss, tr_acc, val_acc))
            log+='\n'
            flog.write(log.format(epoch, tr_loss, val_loss, tr_acc, val_acc))
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), res_dir + "{}.dth".format(exp_name))
                print("Saving model at iteration {}".format(epoch))
                count = 0
            else:
                count += 1
            if count == 200:
                break

            itr_time.append(time.time()-start_itr)

        model.load_state_dict(torch.load(res_dir + "{}.dth".format(exp_name)))
        cr, test_acc, test_f1 = test2(data, model, loader)
        print(cr)
        flog.write(cr+"\n")

        stop_time = time.time() - start_time
        avg_itr_time = np.mean(itr_time)

        flog.write("Avg iteration time: {:.3f} s\n".format(avg_itr_time))
        flog.write("Execution time: {:.3f} s\n".format(stop_time))
        print("Avg iteration time: {:.3f} s\n".format(avg_itr_time))
        print("Execution time: {:.3f} s\n".format(stop_time))


class RedditSage(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(RedditSage, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['../data/RedditSage/embs.trc', '../data/RedditSage/edge_idx.trc', '../data/RedditSage/train_mask.trc', '../data/RedditSage/test_mask.trc', '../data/RedditSage/val_mask.trc', '../data/RedditSage/y.trc']

    @property
    def processed_file_names(self):
        return ['reddit-sage-data.pt']

    def download(self):
        pass

    def process(self):

        #x = torch.load('../data/RedditSage/embs.trc')
        x = torch.Tensor((5,602), dtype=torch.float)
        edge_index = torch.load('../data/RedditSage/edge_idx.trc')
        train_mask = torch.load('../data/RedditSage/train_mask.trc')
        val_mask = torch.load('../data/RedditSage/val_mask.trc')
        test_mask = torch.load('../data/RedditSage/test_mask.trc')
        y = torch.load('../data/RedditSage/y.trc')

        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        torch.save(self.collate(data), self.processed_paths[0])

def main(exps):
    for e in exps:
        exp(e['name'], e['dataset_name'], e['seed'], e['style'], e['shared'])


if __name__ == '__main__':
    exps = [{"name": 'laf_sage_cora2', "dataset_name": 'reddit', "seed": 2603, "style": 'frac', "shared": True},
             ]
    main(exps)


