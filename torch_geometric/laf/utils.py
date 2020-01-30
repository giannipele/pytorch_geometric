import random as rand
import torch
from torch.autograd import Variable
from torch import nn
import numpy as np


class StoppingCriteria:
    import sys
    loss = sys.float_info.max
    accuracy = 0.
    iteration = 0
    counter = 0
    stop = False
    model = nn.Module()

    def update(self, iteration, loss, accuracy, model):
        if loss < self.loss:
            self.loss = loss
            self.iteration = iteration
            self.accuracy = accuracy
            self.model = model
            self.counter = 0
            self.stop = False
        else:
            self.counter += 1
        if self.counter == 100:
            self.stop = True

    def __str__(self):
        return "Stopped at iteration {} at a loss value of {:.3f}".format(self.iteration, self.loss)
    

def create_movielens_data_sets(filename):
    data = {}
    with open(filename, 'r')as fin:
        lines = fin.readlines()
        for l in lines[1:]:
            info = l.strip().split(",")
            bag = [0.0001] * int(info[1]) + [0.9999] * int(info[2])
            t_bag = Variable(torch.FloatTensor(bag), requires_grad=False)
            y = int(info[1]) / int(info[2])
            t_y = Variable(torch.FloatTensor([y]), requires_grad=False)
            data[int(info[0])] = (t_bag, t_y)
    return data


def generate_synth_dataset(examples=100, size=10, target='max'):
    X = []
    var = False
    if size < 0:
        print("Sets size value is negative ({}). The size of the generated sets is uniformly chosen between 1 to {}".format(size, np.abs(size)))
        var = True
        size = np.abs(size)

    if var:
        for i in range(examples):
            s = np.random.randint(low=2, high=size + 1)
            X.append(np.random.uniform(0, 1, size=s))
    else:
        X = [np.random.uniform(0, 1, size=size) for _ in range(examples)]

    y = generate_target_function(X, target)

    return X, y


def generate_bimodal_dataset(mu1=0.3, sigma1=0.1, mu2=0.7, sigma2=0.09, examples=100, size=10, target='max'):
    X = []
    var = False
    if size < 0:
        print("Sets size value is negative ({}). The size of the generated sets is uniformly chosen between 1 to {}".format(size, np.abs(size)))
        var = True
        size = np.abs(size)
    for i in range(examples):
        if var:
            s = np.random.randint(low=2, high=size)
        else:
            s = size
        sample1 = np.array(np.random.normal(loc=mu1, scale=sigma1, size=int(s*0.3)))
        sample2 = np.array(np.random.normal(loc=mu2, scale=sigma2, size=int(s*0.7)))
        sample = np.concatenate((sample1,sample2))
        mi = min(sample)
        sample -= mi
        ma = max(sample)
        if ma == 0:
            ma = 1e-08
        sample /= ma
        np.random.shuffle(sample)
        X.append(sample)
    y = generate_target_function(X, target)

    return X,y


def generate_incremental_dataset(size, target):
    X = []
    if size < 0:
        print("Sets size value is negative ({}). The size of the generated sets is uniformly chosen between 1 to {}".format(size, np.abs(size)))
        size = np.abs(size)
    for i in range(1, size + 1):
        X.append(np.random.uniform(0, 1, size=i))
    y = generate_target_function(X, target)
    return X, y


def generate_target_function(X, target="max"):
    y = []
    if target == 'max':
        y = [[np.max(x)] for x in X]
    elif target == 'min':
        y = [[np.min(x)] for x in X]
    elif target == 'count':
        y = [[len(x)] for x in X]
    elif target == 'mean':
        y = [[np.mean(x)] for x in X]
    elif target == 'sum':
        y = [[np.sum(x)] for x in X]
    elif target == 'minmax':
        for x in X:
            xmax = max(1e-08, np.max(x))
            y.append([np.min(x) / xmax])
    elif target == 'ratio':
        y = [[np.sum(x)/(np.sum(1-x)+1)] for x in X]
    return y

def _non_zero_tensor(self, tensor):
    eps = 1e-6
    ctensor = tensor.clone()
    ctensor[ctensor <= 0] = eps
    return ctensor
