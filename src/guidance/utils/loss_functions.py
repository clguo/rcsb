import torch
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure

class LossFunctionCalculator():
    def __init__(
            self,
            l1_scale: float,
            l2_scale: float,
            tv_scale: float,
            ssim_scale: float,
            rescale: bool = False,
    ):
        self.l1_scale = l1_scale
        self.l2_scale = l2_scale
        self.tv_scale = tv_scale
        self.ssim_scale = ssim_scale
        self.rescale = rescale

        self.ssim = StructuralSimilarityIndexMeasure(data_range=(0., 1.))

    def calculate_loss(
            self,
            output: torch.Tensor,
            target: torch.Tensor,
    ) -> torch.Tensor:
        acc = 0

        if self.l1_scale != 0:
            loss = F.l1_loss(output, target)
            if not torch.isnan(loss):
                if self.rescale:
                    loss = loss.log()
                acc += loss * self.l1_scale
        if self.l2_scale != 0:
            loss = F.mse_loss(output, target)
            if not torch.isnan(loss):
                if self.rescale:
                    loss = (loss + 1e-6).log()
                acc += loss * self.l2_scale
        if self.tv_scale != 0:
            loss = self.total_variation_loss(output)
            if not torch.isnan(loss):
                if self.rescale:
                    loss = (loss + 1e-6).log()
                acc += loss * self.tv_scale
        if self.ssim_scale != 0:
            self.ssim = self.ssim.to(output.device)
            loss = 1 - self.ssim(output, target)
            if not torch.isnan(loss):
                if self.rescale:
                    loss = (loss + 1e-6).log()
                acc += loss * self.ssim_scale
        return acc
    
    def total_variation_loss(self, x: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
               torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        
        # Calculate the number of elements contributing to each sum to approximate normalization
        num_elements = x.shape[1] * (x.shape[2] * (x.shape[3] - 1) + (x.shape[2] - 1) * x.shape[3])
        
        # Normalize the loss to be in the range [0, 1]
        normalized_loss = loss / num_elements
        
        return normalized_loss