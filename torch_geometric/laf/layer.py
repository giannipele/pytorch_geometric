import torch
from torch.nn import Parameter, Module, Sigmoid, Linear, ReLU


class LAFLayer(Module):
    def __init__(self, **kwargs):
        super(LAFLayer, self).__init__()
        if 'parameters' in kwargs.keys():
            self.weights = Parameter(kwargs['parameters'], requires_grad=True)
            self.units = self.weights.shape[1]
        elif 'units' in kwargs.keys():
            self.units = kwargs['units']
            self.weights = Parameter(torch.rand(13, self.units), requires_grad=True)
        else:
            self.units = 1
            self.weights = Parameter(torch.rand(13, self.units), requires_grad=True)
        if 'max' in kwargs.keys():
            self.max = kwargs['max']
        else:
            self.max = 1


class AggregationLayer(LAFLayer):
    def __init__(self, **kwargs):
        super(AggregationLayer, self).__init__(**kwargs)

    def forward(self, data, **kwargs):
        if 'max' in kwargs.keys():
            self.max = kwargs['max']
        eps = 1e-06
        sup = 1e06
        expand_data = []
        try:
            if data.ndim == 1:
                expand_data = data[None, :].expand(self.units, -1)
            elif data.ndim == 2:
                expand_data = data[:,0].expand(self.units, data.shape[0])
        except:
            print("Error in data expansion in aggregation function.")
            print(data, data.shape)

        data_1 = torch.clamp(expand_data.clone(), eps, sup)
        data_2 = torch.clamp(self.max - expand_data.clone(), eps, sup)
        data_3 = torch.clamp(expand_data.clone(), eps, sup)
        data_4 = torch.clamp(self.max - expand_data.clone(), eps, sup)

        exp_1 = torch.pow(data_1, self.weights[1, :, None])
        exp_2 = torch.pow(data_2, self.weights[3, :, None])
        exp_3 = torch.pow(data_3, self.weights[5, :, None])
        exp_4 = torch.pow(data_4, self.weights[7, :, None])

        sum_1 =torch.clamp(torch.sum(exp_1, dim=1), eps, sup)
        sum_2 =torch.clamp(torch.sum(exp_2, dim=1), eps, sup)
        sum_3 =torch.clamp(torch.sum(exp_3, dim=1), eps, sup)
        sum_4 =torch.clamp(torch.sum(exp_4, dim=1), eps, sup)

        sqrt_1 = torch.pow(sum_1, self.weights[0])
        sqrt_2 = torch.pow(sum_2, self.weights[2])
        sqrt_3 = torch.pow(sum_3, self.weights[4])
        sqrt_4 = torch.pow(sum_4, self.weights[6])

        term_1 = sqrt_1 * self.weights[8]
        term_2 = (self.max - sqrt_2) * self.weights[9]
        term_3 = sqrt_3 * self.weights[10]
        term_4 = (self.max - sqrt_4) * self.weights[11]

        num = term_1 + term_2
        den = torch.clamp(term_3 + term_4 + self.weights[12], eps, sup)

        res = num / den

        return res


class ElementAggregationLayer(LAFLayer):
    def __init__(self, **kwargs):
        super(ElementAggregationLayer,self).__init__(**kwargs)

    def forward(self, data, **kwargs):
        if 'max' in kwargs.keys():
            max = kwargs['max']
        else:
            max = self.max
        eps = 1e-6
        sup = 1e6

        data_1 = torch.clamp(data.clone(), eps, sup)
        data_2 = torch.clamp(max - data.clone(), eps, sup)
        data_3 = torch.clamp(data.clone(), eps, sup)
        data_4 = torch.clamp(max - data.clone(), eps, sup)

        exp_1 = torch.pow(data_1, self.weights[1])
        exp_2 = torch.pow(data_2, self.weights[3])
        exp_3 = torch.pow(data_3, self.weights[5])
        exp_4 = torch.pow(data_4, self.weights[7])

        sum_1 = torch.clamp(torch.sum(exp_1, dim=0), eps, sup)
        sum_2 = torch.clamp(torch.sum(exp_2, dim=0), eps, sup)
        sum_3 = torch.clamp(torch.sum(exp_3, dim=0), eps, sup)
        sum_4 = torch.clamp(torch.sum(exp_4, dim=0), eps, sup)

        sqrt_1 = torch.pow(sum_1, self.weights[0])
        sqrt_2 = torch.pow(sum_2, self.weights[2])
        sqrt_3 = torch.pow(sum_3, self.weights[4])
        sqrt_4 = torch.pow(sum_4, self.weights[6])

        term_1 = sqrt_1 * self.weights[8]
        term_2 = (max - sqrt_2) * self.weights[9]
        term_3 = sqrt_3 * self.weights[10]
        term_4 = (max - sqrt_4) * self.weights[11]

        num = term_1 + term_2
        den = torch.clamp(term_3 + term_4 + self.weights[12], eps, sup)

        res = num / den

        return res

