import torch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops

from ..inits import reset
from itertools import repeat
import lhsmdu
from torch_geometric.laf import ElementAggregationLayer, FractionalElementAggregationLayer


class GINConv(MessagePassing):
    r"""The graph isomorphism operator from the `"How Powerful are
    Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right),

    here :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* a MLP.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon` value.
            (default: :obj:`0`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, nn, eps=0, train_eps=False, **kwargs):
        super(GINConv, self).__init__(aggr='add', **kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_index, _ = remove_self_loops(edge_index)
        out = self.nn((1 + self.eps) * x + self.propagate(edge_index, x=x))
        return out

    def message(self, x_j):
        return x_j

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)



class GINLafConv(MessagePassing):
    r"""The graph isomorphism operator from the `"How Powerful are
    Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper

    .. math::
          \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
          \mathbf{x}_i + laf(\{{\mathbf{x}_j : j \in \mathcal{N}(i)} \}) \right),

    here :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* a MLP, and
          math:`laf(X) denotes the aggregation function.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon` value.
            (default: :obj:`0`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, nn, eps=0, train_eps=False, **kwargs):
        self.aggr = 'laf'
        seed = 42
        atype = 'frac'
        shared = False
        embed_dim = '32'
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
        if 'embed_dim' in kwargs.keys():
            embed_dim = int(kwargs['embed_dim'])
            del kwargs['embed_dim']

        super(GINLafConv, self).__init__(aggr='add', **kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

        if shared:
            params = torch.Tensor(lhsmdu.sample(13, 1, randomSeed=seed))
        else:
            params = torch.Tensor(lhsmdu.sample(13, embed_dim, randomSeed=seed))
        #params = torch.Tensor(lhsmdu.sample(13, 1, randomSeed=seed))
        #par = torch.Tensor([[1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0]] * out_channels)
        #par = torch.Tensor([[1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0]])
        #params = par.t()
        if atype == 'minus':
            self.aggregation = ElementAggregationLayer(parameters=params)
        elif atype == 'frac':
            self.aggregation = FractionalElementAggregationLayer(parameters=params)

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_index, _ = remove_self_loops(edge_index)
        out = self.nn((1 + self.eps) * x + self.propagate(edge_index, x=x))
        return out

    def message(self, x_j):
        return x_j

    def aggregate(self, inputs, index, dim_size):  # pragma: no cover
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        By default, delegates call to scatter functions that support
        "add", "mean" and "max" operations specified in :meth:`__init__` by
        the :obj:`aggr` argument.
        """
        return self._laf_aggregate(src=inputs, index=index, dim=self.node_dim, dim_size=dim_size)

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
            # norm_v = torch.clamp(v, 1e-06, 1e06)
            v_min = torch.min(v)
            v_max = torch.max(v)
            # data = (norm_v - v_min)/max((v_max - v_min), 1e-06)
            data = v - v_min
            out[k] = self.aggregation(data, max=torch.max(data))
        return out

    def _get_vectors_to_aggregate(self, src, ids):
        vecs = {}
        for id in ids.keys():
            vecs[id] = src[ids[id]]
        return vecs

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


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







