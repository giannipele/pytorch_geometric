import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import scatter_
from laf.layer import ElementAggregationLayer
from torch_scatter.utils.gen import gen
import lhsmdu

from ..inits import uniform


class SAGELafConv(MessagePassing):
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
        aggr = 'laf'
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
        self.sigmoid = torch.sigmoid
        params = torch.Tensor(lhsmdu.sample(13, out_channels, randomSeed=92))
        #par = torch.Tensor([[1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0]] * out_channels)
        #params = par.t()
        self.aggregation = ElementAggregationLayer(parameters=params)

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
        if self.aggr == 'laf':
            return self._laf_aggregate(src=inputs, index=index, dim=self.node_dim, dim_size=dim_size)
        else:
            return scatter_(self.aggr, inputs, index, self.node_dim, dim_size)

    def _laf_aggregate(self, src, index, dim=-1, out=None, dim_size=None, fill_value=0):
        src, out, index, dim = gen(src, index, dim, out, dim_size, fill_value)
        ids = {}
        for c in range(index.shape[0]):
            id = int(index[c][0])
            if id not in ids.keys():
                ids[id] = []
            ids[id].append(c)

        to_aggregate = self._get_vectors_to_aggregate(src, ids)
        for k, v in to_aggregate.items():
            data = self.sigmoid(v)
            out[k] = self.aggregation(data)
        return out

    def _get_vectors_to_aggregate(self, src, ids):
        vecs = {}
        for id in ids.keys():
            vecs[id] = src[ids[id]]
        return vecs

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
