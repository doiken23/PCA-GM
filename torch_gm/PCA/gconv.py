import torch
import torch.nn as nn
import torch.nn.functional as F


class Gconv(nn.Module):
    """
    (Intra) graph convolution operation, with single convolutional layer
    """
    def __init__(self, in_channels: int, out_channels:int):
        """
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        """
        super(Gconv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.a_fc = nn.Linear(self.in_channels, self.out_channels)
        self.u_fc = nn.Linear(self.in_channels, self.out_channels)

    def forward(self, A, x, norm=True):
        if norm is True:
            A = F.normalize(A, p=1, dim=-2)

        ax = self.a_fc(x)
        ux = self.u_fc(x)
        x = torch.bmm(A, F.relu(ax)) + F.relu(ux)  # has size (bs, N, num_outputs)

        return x


class SiameseGconv(nn.Module):
    """
    Perform graph convolution on two input graphs (g1, g2)
    """
    def __init__(self, in_channels: int, num_features: int):
        """
        in_channels (int): number of input channels
        num_features (int): number of output features
        """
        super(SiameseGconv, self).__init__()
        self.gconv = Gconv(in_channels, num_features)

    def forward(self, g1, g2):
        # embx are tensors of size (bs, N, num_features)
        emb1 = self.gconv(*g1)
        emb2 = self.gconv(*g2)
        return emb1, emb2
