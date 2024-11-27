from typing import Callable, List, Union

import torch
from torch import Tensor

from torch_geometric.nn import GCNConv, GATConv, TopKPooling
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import OptTensor, PairTensor
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    to_torch_csr_tensor,
)
from torch_geometric.utils.repeat import repeat


class GraphGoldNet(torch.nn.Module):
    r"""The Graph U-Net model from the `"Graph U-Nets"
    <https://arxiv.org/abs/1905.05178>`_ paper which implements a U-Net like
    architecture with graph pooling and unpooling operations.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        depth (int): The depth of the U-Net architecture.
        pool_ratios (float or [float], optional): Graph pooling ratio for each
            depth. (default: :obj:`0.5`)
        sum_res (bool, optional): If set to :obj:`False`, will use
            concatenation for integration of skip connections instead
            summation. (default: :obj:`True`)
        act (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.nn.functional.relu`)
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        depth: int,
        pool_ratios: Union[float, List[float]] = 0.5,
        sum_res: bool = True,
        act: Union[str, Callable] = 'relu',
        heads: int = 4,
        dropout: float = 0.6,
    ):
        super().__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth-1
        self.pool_ratios = repeat(pool_ratios, depth)
        self.act = activation_resolver(act)
        self.sum_res = sum_res

        channels = hidden_channels

        self.down_convs = torch.nn.ModuleList()
        self.down_attentions = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()

        self.down_convs.append(GCNConv(in_channels, channels, improved=True))
        self.down_attentions.append(GATConv(in_channels, channels // heads, heads=heads, dropout=dropout))

        for i in range(depth):
            self.pools.append(TopKPooling(channels, self.pool_ratios[i]))
            self.down_convs.append(GCNConv(channels, channels, improved=True))
            self.down_attentions.append(GATConv(channels, channels // heads, heads=heads, dropout=dropout))

        in_channels = channels if sum_res else 2 * channels

        self.up_convs = torch.nn.ModuleList()
        self.up_attentions = torch.nn.ModuleList()

        for i in range(depth - 1):
            self.up_convs.append(GCNConv(in_channels, channels, improved=True))
            self.up_attentions.append(GATConv(channels, channels // heads, heads=heads, dropout=dropout))

        self.last_conv = GCNConv(in_channels, out_channels, improved=True)

        self.reset_parameters()


    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()
        for att in self.down_attentions:
            att.reset_parameters()
        for att in self.up_attentions:
            att.reset_parameters()
        self.last_conv.reset_parameters()


    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor,
                batch: OptTensor = None) -> Tensor:
        """"""  # noqa: D419
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        # edge_weight = x.new_ones(edge_index.size(1))

        _x = self.down_convs[0](x, edge_index, edge_weight)
        _x = self.act(_x)
        x_a = self.down_attentions[0](x + _x, edge_index, edge_weight)
        x_a = self.act(x_a)

        x = _x

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []
        xas = [x_a]
        edge_indices_a = [edge_index]
        edge_weights_a = [edge_weight]
        perms_a = []

        edge_index_a, edge_weight_a, batch_a = edge_index, edge_weight, batch

        for i in range(1, self.depth + 1):
            x_a, edge_index_a, edge_weight_a, batch_a, perm_a, _ = self.pools[i - 1](
                x_a, edge_index_a, edge_weight_a, batch_a)
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](
                x, edge_index, edge_weight, batch)

            x = self.down_convs[i](x, edge_index, edge_weight)
            x = self.act(x)
            x_a = self.down_attentions[i](x + x_a, edge_index_a, edge_weight_a)
            x_a = self.act(x_a)

            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
                xas += [x_a]
                edge_indices_a += [edge_index_a]
                edge_weights_a += [edge_weight_a]
            perms += [perm]
            perms_a += [perm_a]

        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]
            res_a = xas[j]
            edge_index_a = edge_indices_a[j]
            edge_weight_a = edge_weights_a[j]
            perm_a = perms_a[j]

            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up if self.sum_res else torch.cat((res, up), dim=-1)

            up_a = torch.zeros_like(res_a)
            up_a[perm_a] = x_a
            x_a = res_a + up_a if self.sum_res else torch.cat((res_a, up_a), dim=-1)

            x = self.up_convs[i](x, edge_index, edge_weight)
            x = self.act(x)
            x_a = self.up_attentions[i](x + x_a, edge_index_a, edge_weight_a)
            x_a = self.act(x_a)

        x_a = self.last_conv(x + x_a, edge_index_a, edge_weight_a)

        return x_a


    def augment_adj(self, edge_index: Tensor, edge_weight: Tensor,
                    num_nodes: int) -> PairTensor:
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        adj = to_torch_csr_tensor(edge_index, edge_weight,
                                  size=(num_nodes, num_nodes))
        adj = (adj @ adj).to_sparse_coo()
        edge_index, edge_weight = adj.indices(), adj.values()
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.hidden_channels}, {self.out_channels}, '
                f'depth={self.depth}, pool_ratios={self.pool_ratios})')