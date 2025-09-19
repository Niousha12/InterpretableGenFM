from typing import Tuple

import torch
from einops import rearrange
from torch import nn as nn


class IdentityLayer(nn.Module):
    def forward(self, x):
        return x


@torch.no_grad()
def teacher_states(teacher, input_ids, attention_mask):
    # Separate no-grad helper to avoid accidental teacher grads
    out = teacher(input_ids=input_ids, attention_mask=attention_mask,
                  output_all_encoded_layers=True)  # per your teacher call
    # Per your teacher usage: teacher_outputs[0] -> hidden states list/tuple
    return out[0]


class IndexPutFirstAxis(torch.autograd.Function):

    @staticmethod
    def forward(ctx, values: torch.Tensor, indices: torch.Tensor, first_axis_dim) -> torch.Tensor:
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim >= 2
        output = torch.zeros(first_axis_dim, *values.shape[1:], device=values.device, dtype=values.dtype)
        output[indices] = values
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        indices, = ctx.saved_tensors
        grad_values = grad_output[indices]
        return grad_values, None, None


def pad_input(hidden_states: torch.Tensor, indices: torch.Tensor, batch: int, seqlen: int) -> torch.Tensor:
    """Add padding to sequences.

    Arguments:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz)
        batch: int batch_size
        seqlen: int max sequence length

    Returns:
        hidden_states: (batch, seqlen, ...)
    """
    index_put_first_axis = IndexPutFirstAxis.apply
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, '(b s) ... -> b s ...', b=batch)
