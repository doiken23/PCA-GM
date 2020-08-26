import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """
    Cross entropy loss between two permutations.
    """
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(
            self, pred_perm: torch.Tensor, gt_perm: torch.Tensor,
            src_ns: torch.Tensor, tgt_ns: torch.Tensor) -> torch.Tensor:
        """
        pred_perm (torch.Tensor): predicted permutation
        gt_perm (torch.Tensor): ground-truth permutation
        src_ns (torch.Tensor): number of nodes in source graph
        tgt_ns (torch.Tensor): number of nodes in target graph
        """
        batch_num = pred_perm.size(0)

        pred_perm = pred_perm.to(dtype=torch.float32)

        assert torch.all((pred_perm >= 0) * (pred_perm <= 1)), \
            'The value of predicted permutation is invalid'
        assert torch.all((gt_perm >= 0) * (gt_perm <= 1)), \
            'The value of ground-truth permutation is invalid'

        loss = torch.tensor(0., dtype=torch.float32, device=pred_perm.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            loss += F.binary_cross_entropy(
                pred_perm[b, :src_ns[b], :tgt_ns[b]],
                gt_perm[b, :src_ns[b], :tgt_ns[b]],
                reduction='sum')
            n_sum += src_ns[b]

        return loss / n_sum
