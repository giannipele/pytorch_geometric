from __future__ import division
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.laf import ElementAggregationLayer, FractionalElementAggregationLayer

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
        self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
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

        if torch.is_tensor(x):
            x = torch.matmul(x, self.weight)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.weight),
                 None if x[1] is None else torch.matmul(x[1], self.weight))

        return self.propagate(edge_index, size=size, x=x,
                              edge_weight=edge_weight)

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
        shared = False
        if 'aggr' in kwargs.keys():
            aggr = kwargs['aggr']
        if 'seed' in kwargs.keys():
            seed = kwargs['seed']
        if 'style' in kwargs.keys():
            atype = kwargs['style']
        if 'shared' in kwargs.keys():
            shared = kwargs['shared']
        del kwargs['seed']
        del kwargs['style']
        del kwargs['shared']

        super(SAGELafConv, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        #self.sigmoid = torch.sigmoid
        if shared:
            params = torch.Tensor(lhsmdu.sample(13, 1, randomSeed=seed))
        else:
            params = torch.Tensor(lhsmdu.sample(13, out_channels, randomSeed=seed))
        #params = torch.Tensor(lhsmdu.sample(13, 1, randomSeed=seed))
        #par = torch.Tensor([[1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0]] * out_channels)
        #par = torch.Tensor([[1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0]])
        #params = par.t()
        if atype == 'minus': 
            self.aggregation = ElementAggregationLayer(parameters=params)
        elif atype == 'frac':
            self.aggregation = FractionalElementAggregationLayer(parameters=params)

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)
        uniform(self.in_channels, self.bias)

    def forward(self, x, edge_index, edge_weight=None, size=None):
        """"""
        if size is None and torch.is_tensor(x):
            edge_index, edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, 1, x.size(0))

        if torch.is_tensor(x):
            x = torch.matmul(x, self.weight)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.weight),
                 None if x[1] is None else torch.matmul(x[1], self.weight))

        return self.propagate(edge_index, size=size, x=x,
                              edge_weight=edge_weight)

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
        return self._laf_aggregate(src=inputs, index=index, dim=self.node_dim, dim_size=dim_size)
        #else:
        #    return scatter_(self.aggr, inputs, index, self.node_dim, dim_size)


    def _laf_aggregate(self, src, index, dim=-1, out=None, dim_size=None, fill_value=0):
        src, out, index, dim = _gen(src, index, dim, out, dim_size, fill_value)
        ids = {}
        for c in range(index.shape[0]):
            id = int(index[c][0])
            if id not in ids.keys():
                ids[id] = []
            ids[id].append(c)

        to_aggregate = self._get_vectors_to_aggregate(src, ids)
        for k, v in to_aggregate.items():
            #norm_v = torch.clamp(v, 1e-06, 1e06)
            v_min = torch.min(v)
            v_max = torch.max(v)
            #data = (norm_v - v_min)/max((v_max - v_min), 1e-06)
            data = v - v_min
            out[k] = self.aggregation(data, max=torch.max(data))
        return out

    def _get_vectors_to_aggregate(self, src, ids):
        vecs = {}
        for id in ids.keys():
            vecs[id] = src[ids[id]]
        return vecs

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
        
def _gen(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    dim = range(src.dim())[dim]  # Get real dim value.

    # Automatically expand index tensor to the right dimensions.
    if index.dim() == 1:
        index_size = list(repeat(1, src.dim()))
        index_size[dim] = src.size(dim)
        if index.numel() > 0:
            index = index.view(index_size).expand_as(src)
        else:  # pragma: no cover
            # PyTorch has a bug when view is used on zero-element tensors.
            index = src.new_empty(index_size, dtype=torch.long)

    # Broadcasting capabilties: Expand dimensions to match.
    if src.dim() != index.dim():
        raise ValueError(
            ('Number of dimensions of src and index tensor do not match, '
             'got {} and {}').format(src.dim(), index.dim()))

    expand_size = []
    for s, i in zip(src.size(), index.size()):
        expand_size += [-1 if s == i and s != 1 and i != 1 else max(i, s)]
    src = src.expand(expand_size)
    index = index.expand_as(src)

    # Generate output tensor if not given.
    if out is None:
        out_size = list(src.size())
        dim_size = maybe_dim_size(index, dim_size)
        out_size[dim] = dim_size
        out = src.new_full(out_size, fill_value)

    return src, out, index, dim


def maybe_dim_size(index, dim_size=None):
    if dim_size is not None:
        return dim_size
    return index.max().item() + 1 if index.numel() > 0 else 0


