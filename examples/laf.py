#import os
#import torch
#import torch.nn.functional as F
#from torch_geometric.datasets import Planetoid
#import torch_geometric.transforms as T
##from torch_geometric.nn import SAGELafConv
#from torch_geometric.nn import SAGEConv, SAGELafConv
#import sys
#import pdb
#import traceback
#from colorama import Fore, Back, Style
##dataset = 'Cora'
##path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
##dataset = Planetoid(path, dataset, T.NormalizeFeatures())
##data = dataset[0]
##
##data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
##data.train_mask[:data.num_nodes - 1000] = 1
##data.val_mask = None
##data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
##data.test_mask[data.num_nodes - 500:] = 1
#
#
#class SAGENet(torch.nn.Module):
#
#    def __init__(self, dataset, **kwargs):
#        seed = 42
#        if 'seed' in kwargs.keys():
#            seed = kwargs['seed']
#        super(SAGENet, self).__init__()
#        self.conv1 = SAGELafConv(dataset.num_features, 16, seed=seed)
#        self.conv2 = SAGELafConv(16, dataset.num_classes, seed=2*seed)
#        #self.conv1 = SAGEConv(dataset.num_features, 16)
#        #self.conv2 = SAGEConv(16, dataset.num_classes)
#
#    def forward(self, data):
#        x, edge_index = data.x, data.edge_index
#        x = F.relu(self.conv1(x, edge_index))
#        x = F.dropout(x, training=self.training)
#        x = self.conv2(x, edge_index)
#        return F.log_softmax(x, dim=1)
#
#
#EPOCH = 500
#
#
#def train(model, data, optimizer):
#    model.train()
#    optimizer.zero_grad()
#    F.nll_loss(model(data)[data.train_mask], data.y[data.train_mask]).backward()
#    optimizer.step()
#    #par = []
#    #for p in model.conv1.aggregation.parameters():
#    #    par.append(p)
#    #print(par)
#
#
#def validate(model, data):
#    model.eval()
#    logits, accs = model(data), []
#    for _, mask in data('train_mask', 'val_mask'):
#        pred = logits[mask].max(1)[1]
#        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
#        accs.append(acc)
#    return accs
#
#
#def test(model, data):
#    model.eval()
#    logits, accs = model(data), []
#    for _, mask in data('train_mask', 'test_mask'):
#        pred = logits[mask].max(1)[1]
#        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
#        accs.append(acc)
#    return accs
#
#
#def main(exp_name):
#    torch.manual_seed(42)
#    dataset = 'Cora'
#    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', dataset)
#    dataset = Planetoid(path, dataset, T.NormalizeFeatures())
#    data = dataset[0]
#    print(data) 
#    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
#    data.train_mask[:data.num_nodes - 1000] = 1
#
#    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
#    data.val_mask[data.num_nodes - 1000: data.num_nodes - 500] = 1
#
#    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
#    data.test_mask[data.num_nodes - 500:] = 1
#
#    print('Train: {}'.format(torch.sum(data.train_mask)))
#    print('Validation: {}'.format(torch.sum(data.val_mask)))
#    print('Test: {}'.format(torch.sum(data.test_mask)))
#
#    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#    data = data.to(device)
#    model = SAGENet(dataset).to(device)
#    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
#    anomaly_detector = GuruMeditation()
#    with anomaly_detector:
#        with open('{}.log'.format(exp_name), 'w') as flog:
#            best_acc = 0
#            for epoch in range(1, EPOCH):
#                #print('Training Model...')
#                train(model, data, optimizer)
#                #print('Validating Model...')
#                train_accs = validate(model, data)
#                log = 'Epoch: {:03d}, Train: {:.4f}, Validation: {:.4f}'
#                print(log.format(epoch, *train_accs))
#                log+='\n'
#                flog.write(log.format(epoch, *train_accs))
#                if train_accs[1] > best_acc:
#                    best_acc = train_accs[1]
#                    torch.save(model.state_dict(), "{}.dth".format(exp_name))
#                    print("Saving model at iteration {}".format(epoch))
#            model.load_state_dict(torch.load("{}.dth".format(exp_name)))
#            accs = test(model, data)
#            print('Test Accuracy: {}'.format(accs[1]))
#            flog.write('Test Accuracy: {}\n'.format(accs[1]))
#
#
#class GuruMeditation (torch.autograd.detect_anomaly):
#    def __init__(self):
#        super(GuruMeditation, self).__init__()
#
#    def __enter__(self):
#        super(GuruMeditation, self).__enter__()
#        return self
#
#    def __exit__(self, type, value, trace):
#        super(GuruMeditation, self).__exit__()
#        if isinstance(value, RuntimeError):
#            traceback.print_tb(trace)
#            halt(str(value))
#
#def halt(msg):
#    print (Fore.RED + "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
#    print (Fore.RED + "┃ Software Failure. Press left mouse button to continue ┃")
#    print (Fore.RED + "┃        Guru Meditation 00000004, 0000AAC0             ┃")
#    print (Fore.RED + "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
#    print(Style.RESET_ALL)
#    print (msg)
#    pdb.set_trace()
#
#if __name__ == '__main__':
#    exp = sys.argv[1]
#    seed = sys.argv[2]
#    main(exp, seed)
#

import os
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import SplineConv, SAGELafConv
import sys
import pdb
import traceback
from colorama import Fore, Back, Style


class SAGENet(torch.nn.Module):
    def __init__(self, dataset, seed):
        super(SAGENet, self).__init__()
        self.conv1 = SAGELafConv(dataset.num_features, 16, seed=seed)
        self.conv2 = SAGELafConv(16, dataset.num_classes, seed=seed)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
EPOCH = 300


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

def main(exp_name, seed):
    torch.manual_seed(seed)
    dataset = 'Cora'
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', dataset)
    dataset = Planetoid(path, dataset, T.NormalizeFeatures())
    data = dataset[0]
    #print(data)
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[:data.num_nodes - 1000] = 1

    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask[data.num_nodes - 1000: data.num_nodes - 500] = 1

    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask[data.num_nodes - 500:] = 1


    print('Train: {}'.format(torch.sum(data.train_mask)))
    print('Validation: {}'.format(torch.sum(data.val_mask)))
    print('Test: {}'.format(torch.sum(data.test_mask)))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    model = SAGENet(dataset, seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    anomaly_detector = GuruMeditation()
    with anomaly_detector:
        with open('{}.log'.format(exp_name), 'w') as flog:
            best_acc = 0
            count = 0
            for epoch in range(1, EPOCH):
                #print('Training Model...')
                train(model, data, optimizer)
                #print('Validating Model...')
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
                if count == 80:
                    break
            model.load_state_dict(torch.load("{}.dth".format(exp_name)))
            accs = test(model, data)
            print('Test Accuracy: {}'.format(accs[1]))
            flog.write('Test Accuracy: {}\n'.format(accs[1]))

class GuruMeditation (torch.autograd.detect_anomaly):
    def __init__(self):
        super(GuruMeditation, self).__init__()
    def __enter__(self):
        super(GuruMeditation, self).__enter__()
        return self
    def __exit__(self, type, value, trace):
        super(GuruMeditation, self).__exit__()
        if isinstance(value, RuntimeError):
            traceback.print_tb(trace)
            halt(str(value))

def halt(msg):
    print (Fore.RED + "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
    print (Fore.RED + "┃ Software Failure. Press left mouse button to continue ┃")
    print (Fore.RED + "┃        Guru Meditation 00000004, 0000AAC0             ┃")
    print (Fore.RED + "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
    print(Style.RESET_ALL)
    print (msg)
    pdb.set_trace()

if __name__ == '__main__':
    exp = sys.argv[1]
    seed = int(sys.argv[2])
    main(exp, seed)


