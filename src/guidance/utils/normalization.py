import torch

import logging
log = logging.getLogger(__name__)

def l2_normalize_gradient(grad, small_const=1e-22):
    grad_norm = grad.view(grad.shape[0], -1).norm(p=2, dim=1).view(grad.shape[0], 1, 1, 1)
    grad_norm = torch.where(grad_norm < small_const, grad_norm + small_const, grad_norm)
    gradient_normalized = grad / grad_norm
    return gradient_normalized


class AdaptiveNormalizer(torch.nn.Module):
    def __init__(self, target_scale: float = 1.):
        super().__init__()
        self.prev_t: int = 0
        self.target_scale = target_scale
    
    def __call__(self, grad: torch.Tensor, t: int) -> torch.Tensor:
        if self.prev_t < t:  # Update after new inpainting starts
            bs, _, _, _ = grad.shape
            max_grad_norm_tensor = grad.view(bs, -1).norm(p=2, dim=1).detach().float()
            self.register_buffer("grad_norm", max_grad_norm_tensor.view(bs, 1, 1, 1))
            log.info("Resetting adaptive normalization")

        assert "grad_norm" in self._buffers
        self.prev_t = t
        return grad / self.grad_norm * self.target_scale
        