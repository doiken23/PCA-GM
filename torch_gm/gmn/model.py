import torch.nn as nn
from torch_gm.gmn.affinity_layer import Affinity
from torch_gm.gmn.displacement_layer import Displacement
from torch_gm.gmn.power_iteration import PowerIteration
from torch_gm.utils.build_graphs import reshape_edge_feature
from torch_gm.utils.fgm import construct_m
from torch_gm.utils.sinkhorn import Sinkhorn
from torch_gm.utils.voting_layer import Voting


class GMNLayer(nn.Module):
    def __init__(
            self, node_feature_dim,
            pi_max_iter, pi_stop_thresh, sh_max_iter, sh_epsilon, v_alpha,
            voting=True, bi_stochastic=True):
        super(GMNLayer, self).__init__()

        # voting and bi_stochastic
        self.voting = voting
        self.bi_stochastic = bi_stochastic

        # define layers
        self.affinity_layer = Affinity(node_feature_dim // 2)
        self.power_iteration = PowerIteration(max_iter=pi_max_iter, stop_thresh=pi_stop_thresh)
        self.bi_stochastic_layer = Sinkhorn(max_iter=sh_max_iter, epsilon=sh_epsilon)
        self.voting_layer = Voting(alpha=v_alpha)
        self.displacement_layer = Displacement()
        self.l2norm = nn.LocalResponseNorm(
            node_feature_dim,
            alpha=node_feature_dim, beta=0.5, k=0
        )

    def forward(
            self, U_src, U_tgt,
            F_src, F_tgt, G_src, G_tgt, H_src, H_tgt, ns_src, ns_tgt, K_G, K_H,
            P_src=None, P_tgt=None):
        """

        Args:
            U_src (torch.Tensor): node features of source graph (shape: [b, node_number, feature_dim])
            U_tgt (torch.Tensor): node features of target graph (shape: [b, node_number, feature_dim])
            F_src (torch.Tensor): edge features of source graph (shape: [b, edge_number, feature_dim])
            F_tgt (torch.Tensor): edge features of target graph (shape: [b, edge_number, feature_dim])
            G_src (torch.Tensor): node-edge incidence matrix (shape:[b, node_number, edge_number])
            G_tgt (torch.Tensor): node-edge incidence matrix (shape:[b, node_number, edge_number])
            H_src (torch.Tensor): node-edge incidence matrix (shape:[b, node_number, edge_number])
            H_tgt (torch.Tensor): node-edge incidence matrix (shape:[b, node_number, edge_number])
            ns_src (int): number of source node
            ns_tgt (int): number of target node
            K_G (torch.Tensor): kronecker product of G2, G1
            K_H (torch.Tensor): kronecker product of H2, H1
            P_src (torch.Tensor, optional): position of source node
            P_tgt (torch.Tensor, optional): position of target node

        Returns:
            s (torch.Tensor): confidence matrix
            d (torch.Tensor, optional): displacement vector
        """

        # construct edge features
        X = reshape_edge_feature(F_src, G_src, H_src)
        Y = reshape_edge_feature(F_tgt, G_tgt, H_tgt)

        # construct affinity matrix
        Me, Mp = self.affinity_layer(X, Y, U_src, U_tgt)
        M = construct_m(Me, Mp, K_G, K_H)

        # solve assignment problem
        v = self.power_iteration(M)
        s = v.view(v.size(0), U_tgt.size(1), U_src.size(1)).transpose(1, 2)

        # make double-stochastic matrix
        if self.bi_stochastic:
            s = self.bi_stochastic_layer(s, ns_src, ns_tgt)

        # convert confidence-maps to displacements
        if self.voting:
            s = self.voting_layer(s, ns_src, ns_tgt)
            if (P_src is not None) and (P_tgt is not None):
                d, _ = self.displacement_layer(s, P_src, P_tgt)
            return s, d
        else:
            return s
