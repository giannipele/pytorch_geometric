from __future__ import division
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.laf import ElementAggregationLayer, FractionalElementAggregationLayer, ScatterAggregationLayer

from itertools import repeat
#import laf

from ..inits import uniform
import lhsmdu

class SAGEConv(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    .. math::
        \mathbf{\hat{x}}_i &= \mathbf{\Theta} \cdot
        \mathrm{mean}_{j \in \mathcal{N(i) \cup \{ i \}}}(\mathbf{x}_j)

        \mathbf{x}^{\prime}_i &= \frac{\mathbf{\hat{x}}_i}
        {\| \mathbf{\hat{x}}_i \|_2}.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, normalize=False, bias=True,
                 **kwargs):
        super(SAGEConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(in_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)
        uniform(self.in_channels, self.bias)

    def forward(self, x, edge_index, edge_weight=None, size=None):
        """"""
        if size is None and torch.is_tensor(x):
            edge_index, edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, 1, x.size(0))

        x = self.propagate(edge_index, size=size, x=x,
                           edge_weight=edge_weight)
        if torch.is_tensor(x):
            x = torch.matmul(x, self.weight)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.weight),
                 None if x[1] is None else torch.matmul(x[1], self.weight))

        return F.relu(x)

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        if self.normalize:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class SAGEConvCorrect(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    .. math::
        \mathbf{\hat{x}}_i &= \mathbf{\Theta} \cdot
        \mathrm{mean}_{j \in \mathcal{N(i) \cup \{ i \}}}(\mathbf{x}_j)

        \mathbf{x}^{\prime}_i &= \frac{\mathbf{\hat{x}}_i}
        {\| \mathbf{\hat{x}}_i \|_2}.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, normalize=False, bias=True,
                 **kwargs):
        super(SAGEConvCorrect, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(in_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)
        uniform(self.in_channels, self.bias)

    def forward(self, x, edge_index, edge_weight=None, size=None):
        """"""
        if size is None and torch.is_tensor(x):
            edge_index, edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, 1, x.size(0))

        x = self.propagate(edge_index, size=size, x=x,
                              edge_weight=edge_weight)
        if torch.is_tensor(x):
            x = torch.matmul(x, self.weight)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.weight),
                 None if x[1] is None else torch.matmul(x[1], self.weight))

        return F.relu(x)

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        if self.normalize:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class SAGELafConv(MessagePassing):

    def __init__(self, in_channels, out_channels, normalize=False, bias=True,
                 **kwargs):
        aggr = 'laf'
        seed = 42
        atype = 'frac'
        shared = True
        fun = 'mean'
        if 'aggr' in kwargs.keys():
            aggr = kwargs['aggr']
        if 'seed' in kwargs.keys():
            seed = kwargs['seed']
            del kwargs['seed']
        if 'style' in kwargs.keys():
            atype = kwargs['style']
            del kwargs['style']
        if 'shared' in kwargs.keys():
            shared = kwargs['shared']
            del kwargs['shared']
        if 'fun' in kwargs.keys():
            fun = kwargs['fun']
            del kwargs['fun']

        super(SAGELafConv, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(in_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        #if shared:
        #    params = torch.rand((13, 1))
        #else:
        #    params = torch.Tensor(lhsmdu.sample(13, out_channels, randomSeed=seed))
        if fun == 'mean':
            params = torch.Tensor([[1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0]]).t()
        if fun == 'sum':
            params = torch.Tensor([[1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]]).t()
        if fun == 'max':
            params = torch.Tensor([[10, 10, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]]).t()
        if fun == 'min':
            params = torch.Tensor([[0, 0, 10, 10, 0, 0, 0, 0, 0, 1, 0, 0, 1]]).t()
        if fun == 'minmax':
            params = torch.Tensor([[0, 0, 10, 10, 10, 10, 0, 0, 0, 1, 1, 0, 0]]).t()
        if fun == 'count':
            params = torch.Tensor([[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]]).t()
        if fun == 'maxmin':
            params = torch.Tensor([[10, 10, 0, 0, 0, 0, 10, 10, 1, 0, 0, 1, 0]]).t()
        eps = torch.randint(-1000,1000,(13,1), dtype=torch.float)/100000 
        params = params + eps
        print(params)
        if atype == 'minus':
            self.aggregation = ElementAggregationLayer(parameters=params)
        elif atype == 'frac':
            #self.aggregation = FractionalElementAggregationLayer(parameters=params)
            self.aggregation = ScatterAggregationLayer(parameters=params)

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)
        uniform(self.in_channels, self.bias)

    def forward(self, x, edge_index, edge_weight=None, size=None):
        """"""
        if size is None and torch.is_tensor(x):
            edge_index, edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, 1, x.size(0))

        x = self.propagate(edge_index, size=size, x=x,
                              edge_weight=edge_weight)
        if torch.is_tensor(x):
            x = torch.matmul(x, self.weight)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.weight),
                 None if x[1] is None else torch.matmul(x[1], self.weight))
        return x

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        if self.normalize:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        return aggr_out

    def aggregate(self, inputs, index, dim_size):  # pragma: no cover
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        By default, delegates call to scatter functions that support
        "add", "mean" and "max" operations specified in :meth:`__init__` by
        the :obj:`aggr` argument.
        """

        #out = max_pool(index,inputs)
        v_min = torch.min(inputs)
        inputs = inputs - v_min
        out = self.aggregation(inputs, index)
        return out


