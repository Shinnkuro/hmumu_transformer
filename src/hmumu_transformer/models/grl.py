from __future__ import annotations

import torch

class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None

def grad_reverse(x: torch.Tensor, lambda_: float) -> torch.Tensor:
    """Gradient reversal layer (GRL)."""
    return _GRL.apply(x, lambda_)
