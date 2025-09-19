import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.helpers import pad_input


class FocalLoss(nn.Module):
    """
    Binary focal loss (BCE-style) for logits of shape [B, 1] or [B].
    targets must be floats in {0,1}.
    """
    def __init__(self, alpha=0.95, gamma=2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = float(alpha)  # weight for the positive class
        self.gamma = float(gamma)
        assert reduction in {"none", "mean", "sum"}
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        # Ensure shapes match [B, 1]
        if logits.dim() == 1:
            logits = logits.unsqueeze(-1)  # [B] -> [B,1]
        targets = targets.to(dtype=logits.dtype)  # float32
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)  # [B] -> [B,1]

        # BCE with logits per example
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')  # [B,1]

        # p_t: predicted prob of the true class (more stable than mixing formula)
        p = torch.sigmoid(logits)  # [B,1]
        pt = torch.where(targets > 0.5, p, 1 - p)  # [B,1]

        # alpha weighting: alpha for positives, (1-alpha) for negatives
        alpha_t = torch.where(targets > 0.5, torch.full_like(targets, self.alpha),
                              torch.full_like(targets, 1.0 - self.alpha))  # [B,1]

        # focal modulation
        loss = alpha_t * (1 - pt).pow(self.gamma) * bce  # [B,1]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss  # [B,1]


def kd_loss_from_states(
        teacher_hidden_states,
        student_hidden_states,
        attention_mask,
        B,
        C,
        L,
        layer_map,
        criterion,
        enable_logits=False
):
    losses = []
    logits = []
    idx = torch.nonzero(attention_mask.reshape(-1), as_tuple=False).reshape(-1).to(teacher_hidden_states[0].device)

    for t_layer, s_layer in layer_map.items():
        t_hid = teacher_hidden_states[t_layer]  # (nnz, H) or (nnz, hidden)
        s_hid = student_hidden_states[s_layer]  # (B*C*L, H) but already padded by HF

        # teacher is packed by valid tokens -> pad back to (B*C, L, H)
        t_hid = pad_input(t_hid, idx, B * C, L)

        # reshape to (B, C, L, H)
        t_hid = t_hid.reshape(B, C, L, -1)
        s_hid = s_hid.reshape(B, C, L, -1)

        # ref/alt deltas
        delta_t = t_hid[:, 0, :, :] - t_hid[:, 1, :, :]  # (B, L, H)
        delta_s = s_hid[:, 0, :, :] - s_hid[:, 1, :, :]  # (B, L, H)

        layer_loss = criterion(delta_s, delta_t)
        losses.append(layer_loss)

        if enable_logits:
            layer_logit = ((delta_t - delta_s) ** 2).mean(dim=(1, 2))
            logits.append(layer_logit)

    loss = torch.stack(losses).mean()
    if enable_logits:
        logits = torch.stack(logits).mean(dim=0)
        return loss, logits
    return loss
