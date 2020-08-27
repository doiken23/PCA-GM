#!/usr/bin/env python3

import pytest

import torch

from torch_gm.utils.permutation_loss import CrossEntropyLoss


def test_permutation_loss():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # set loss layer
    criterion = CrossEntropyLoss().to(device)

    # prepare permutations
    pred_perm = torch.sigmoid(torch.randn(10, 64, 64, device=device))
    gt_perm = torch.rand(10, 64, 64, device=device)

    # number of nodes
    src_ns = torch.arange(64, 54, -1, dtype=torch.int64, device=device)
    tgt_ns = torch.arange(55, 65, dtype=torch.int64, device=device)

    # compute loss
    loss = criterion(pred_perm, gt_perm, src_ns, tgt_ns)

    assert type(loss) == torch.Tensor, 'type of loss is wrong'
    assert loss.dtype == torch.float32, 'dtype of loss is wrong'
    assert loss.device == device, 'device of loss is wrong'
    assert loss.size() == torch.Size(), 'size of loss is wrong'
