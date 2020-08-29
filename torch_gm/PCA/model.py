from typing import Any, Optional

import torch
import torch.nn as nn
from torch_gm.pca.affinity_layer import Affinity
from torch_gm.pca.gconv import SiameseGconv
from torch_gm.utils.sinkhorn import Sinkhorn
from torch_gm.utils.voting_layer import Voting


class PCAGMNet(nn.Module):
    def __init__(
            self,
            gnn_num: int, in_channels: int, gnn_channels: int,
            sinkhorn_iter_num: int, sinkhorn_epsilon: float,
            voting_alpha: float, voting_thresh: Optional[float] = None
            ):
        """
        Arguments:
            gnn_num (int): number of GNN layer
            in_channels (int): number of input node feature channels
            gnn_channels (int): number of gnn feature channels
            sinkhorn_iter_num (int): iteration number of Sinkhorn layer
            sinkhorn_epsilon (float): epsilon number of Sinkhorn layer
            voting_alpha, voting_thresh (float): hyper parameter of voting layer
        """
        super(PCAGMNet, self).__init__()

        self.bi_stochastic = Sinkhorn(max_iter=sinkhorn_iter_num, epsilon=sinkhorn_epsilon)
        self.voting_layer = Voting(alpha=voting_alpha, pixel_thresh=voting_thresh)
        self.gnn_num = gnn_num
        for i in range(self.gnn_num):
            if i == 0:
                gnn_layer = SiameseGconv(in_channels * 2, gnn_channels)
            else:
                gnn_layer = SiameseGconv(gnn_channels, gnn_channels)
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)
            self.add_module('affinity_{}'.format(i), Affinity(gnn_channels))
            if i == self.gnn_num - 2:  # only second last layer will have cross-graph module
                self.add_module(
                    'cross_graph_{}'.format(i), nn.Linear(2 * gnn_channels, gnn_channels))

    def forward(
            self, node_src: torch.Tensor, node_tgt: torch.Tensor,
            G_src: Any, G_tgt: Any, H_src: Any, H_tgt: Any, ns_src:torch.Tensor, ns_tgt: torch.Tensor
            ) -> torch.Tensor:
        """
        Arguments:
            node_src, node_tgt (torch.Tensor): node feature tensor
            G_src, G_tgt, H_src, H_tgt: edge tensors
            ns_src, ns_tgt (torch.Tensor): node number
        Returns:
            s (torch.Tensor): permutation matrix
        """
        # adjacency matrices
        A_src = torch.bmm(G_src, H_src.transpose(1, 2))
        A_tgt = torch.bmm(G_tgt, H_tgt.transpose(1, 2))

        for i in range(self.gnn_num):
            gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
            node_src, node_tgt = gnn_layer([A_src, node_src], [A_tgt, node_tgt])
            affinity = getattr(self, 'affinity_{}'.format(i))
            s = affinity(node_src, node_tgt)
            s = self.voting_layer(s, ns_src, ns_tgt)
            s = self.bi_stochastic(s, ns_src, ns_tgt)

            if i == self.gnn_num - 2:
                cross_graph = getattr(self, 'cross_graph_{}'.format(i))
                node_src_new = cross_graph(
                    torch.cat((node_src, torch.bmm(s, node_tgt)), dim=-1))
                node_tgt_new = cross_graph(
                    torch.cat((node_tgt, torch.bmm(s.transpose(1, 2), node_src)), dim=-1))
                node_src = node_src_new
                node_tgt = node_tgt_new

        return s
