"""
LARS: Layer-wise Adaptive Rate Scaling (You, Gitman & Ginsburg, 2017).

PyTorch does not ship LARS, so here it is. It is the optimizer the large-batch
slide compares against plain linear LR scaling with warmup.

WHY IT EXISTS
-------------
When the global batch gets large, a single global learning rate stops working
well. Different layers have very different ratios of weight magnitude to
gradient magnitude, so one learning rate is simultaneously too big for some
layers and too small for others. Training either diverges or stalls.

LARS computes a SEPARATE learning rate for every layer, from the ratio

    local_lr = trust_coefficient * ||w|| / (||grad|| + weight_decay * ||w||)

so a layer whose weights are large relative to its gradients takes a bigger
step, and vice versa. The global learning rate then scales all of these
together. That is what lets batch sizes in the thousands train stably.

TWO DETAILS THAT MATTER
-----------------------
1. Bias and normalization parameters are EXCLUDED from the adaptation and from
   weight decay. Their norms are tiny and the ratio above becomes meaningless,
   which is the usual way a LARS implementation goes wrong. Build the param
   groups with `exclude_from_lars` below.
2. If either norm is zero (a freshly initialized bias, or a layer with no
   gradient this step) the local rate falls back to 1.0 rather than dividing
   by zero.

USAGE
-----
    from lars import LARS, build_lars_param_groups
    groups = build_lars_param_groups(model, weight_decay=5e-4)
    opt = LARS(groups, lr=base_lr, momentum=0.9, trust_coefficient=0.001)

train_asl.py exposes this as `--optimizer lars`.
"""

import torch
from torch.optim.optimizer import Optimizer


def exclude_from_lars(name, param):
    """
    True for parameters that should not get layer-wise adaptation or weight
    decay: biases and all normalization scales/shifts. They are 1-D.
    """
    return param.ndim <= 1 or name.endswith(".bias")


def build_lars_param_groups(model, weight_decay=5e-4):
    """Split a model's parameters into the adapted group and the excluded group."""
    adapt, exclude = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (exclude if exclude_from_lars(name, p) else adapt).append(p)
    return [
        {"params": adapt, "weight_decay": weight_decay, "lars_adapt": True},
        {"params": exclude, "weight_decay": 0.0, "lars_adapt": False},
    ]


class LARS(Optimizer):
    """
    SGD with momentum plus a per-layer adaptive learning rate.

    Args:
        params: iterable of parameters or param groups. A group may set
                "lars_adapt": False to opt out of the adaptation (use this for
                biases and norm parameters).
        lr: the global learning rate. Scale this with batch size as usual.
        momentum: heavy-ball momentum, 0.9 in the paper.
        weight_decay: applied inside the adaptation, as the paper specifies.
        trust_coefficient: eta in the paper, 0.001 is the standard value.
        eps: guards the division.
    """

    def __init__(self, params, lr=1.0, momentum=0.9, weight_decay=0.0,
                 trust_coefficient=0.001, eps=1e-8):
        if lr < 0.0:
            raise ValueError(f"invalid lr: {lr}")
        if momentum < 0.0:
            raise ValueError(f"invalid momentum: {momentum}")
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        trust_coefficient=trust_coefficient, eps=eps,
                        lars_adapt=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            eta = group["trust_coefficient"]
            eps = group["eps"]
            adapt = group.get("lars_adapt", True)

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad

                if adapt:
                    w_norm = torch.norm(p)
                    g_norm = torch.norm(d_p)
                    # Zero norms make the ratio meaningless; fall back to 1.0.
                    denom = g_norm + wd * w_norm + eps
                    local_lr = torch.where(
                        (w_norm > 0) & (g_norm > 0),
                        eta * w_norm / denom,
                        torch.ones_like(w_norm),
                    )
                    d_p = d_p.add(p, alpha=wd).mul(local_lr)
                elif wd != 0:
                    d_p = d_p.add(p, alpha=wd)

                if momentum != 0:
                    state = self.state[p]
                    buf = state.get("momentum_buffer")
                    if buf is None:
                        buf = torch.clone(d_p).detach()
                        state["momentum_buffer"] = buf
                    else:
                        buf.mul_(momentum).add_(d_p)
                    d_p = buf

                p.add_(d_p, alpha=-lr)

        return loss
