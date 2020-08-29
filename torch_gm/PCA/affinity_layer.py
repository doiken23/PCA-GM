import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class Affinity(nn.Module):
    """
    Affinity Layer to compute the affinity matrix from feature space.
    M = X * A * Y^T
    """
    def __init__(self, d: int):
        """
        Arguments:
            d (int): size of affinity matrix
        """
        super(Affinity, self).__init__()
        self.d = d
        self.A = Parameter(torch.Tensor(self.d, self.d))
        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1. / math.sqrt(self.d)
        self.A.data.uniform_(-stdv, stdv)
        self.A.data += torch.eye(self.d)

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            X (torch.Tensor): feature of graph 1
            Y (torch.Tensor): feature of graph 2
        Returns:
            M (torch.Tensor): affinity matrix
        """
        assert X.shape[2] == Y.shape[2] == self.d
        M = torch.matmul(X, (self.A + self.A.transpose(0, 1)) / 2)  # A should be symmetry
        M = torch.matmul(M, Y.transpose(1, 2))
        return M
