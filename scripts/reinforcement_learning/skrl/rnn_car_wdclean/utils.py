"""Monitoring helpers and running normalizer -- extracted from train_rnn_car_wdclip.py."""

import torch


# WD module loss dim -> human-readable name mapping
_AUX_DIM_NAMES = {
    "module_feture_0_loss": "aux/near1_x_loss",
    "module_feture_1_loss": "aux/near1_y_loss",
    "module_feture_2_loss": "aux/near1_d_loss",
    "module_feture_3_loss": "aux/near2_x_loss",
    "module_feture_4_loss": "aux/near2_y_loss",
    "module_feture_5_loss": "aux/near2_d_loss",
}


def _param_l2_norm(params) -> float:
    """L2 norm of a flat parameter vector."""
    total = 0.0
    for p in params:
        total += p.data.norm(2).item() ** 2
    return total ** 0.5


def _grad_l2_norm(params) -> float:
    """L2 norm of gradients. Returns 0 if no grad exists."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


def _scale_grads(params, scale: float):
    """Scale gradients in-place for a parameter group."""
    if scale >= 1.0:
        return
    for p in params:
        if p.grad is not None:
            p.grad.data.mul_(scale)


def _snapshot_params(params) -> torch.Tensor:
    """Flatten and clone all parameters into a single vector."""
    return torch.cat([p.data.reshape(-1).clone() for p in params])


def _param_delta_norm(before: torch.Tensor, after_params) -> float:
    """L2 norm of (after - before) parameter vector."""
    after = torch.cat([p.data.reshape(-1) for p in after_params])
    return (after - before).norm(2).item()


class RunningNormalizer:
    """Welford's online algorithm for running mean/var normalization."""

    def __init__(self, shape, device, clip=5.0):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = 1e-4
        self.clip = clip

    @torch.no_grad()
    def update(self, x):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x):
        return torch.clamp((x - self.mean) / (self.var.sqrt() + 1e-8), -self.clip, self.clip)
