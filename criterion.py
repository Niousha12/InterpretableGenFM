import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Binary focal loss (BCE-style) for logits of shape [B, 1] or [B].
    targets must be floats in {0,1}.
    """
    def __init__(self, alpha=0.95, gamma=2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = float(alpha)   # weight for the positive class
        self.gamma = float(gamma)
        assert reduction in {"none", "mean", "sum"}
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        # Ensure shapes match [B, 1]
        if logits.dim() == 1:
            logits = logits.unsqueeze(-1)          # [B] -> [B,1]
        targets = targets.to(dtype=logits.dtype)    # float32
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)        # [B] -> [B,1]

        # BCE with logits per example
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')  # [B,1]

        # p_t: predicted prob of the true class (more stable than mixing formula)
        p = torch.sigmoid(logits)                                # [B,1]
        pt = torch.where(targets > 0.5, p, 1 - p)                # [B,1]

        # alpha weighting: alpha for positives, (1-alpha) for negatives
        alpha_t = torch.where(targets > 0.5,
                              torch.full_like(targets, self.alpha),
                              torch.full_like(targets, 1.0 - self.alpha))  # [B,1]

        # focal modulation
        loss = alpha_t * (1 - pt).pow(self.gamma) * bce          # [B,1]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss  # [B,1]
