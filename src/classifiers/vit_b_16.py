from torch import Tensor
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights
from .base import ClassifierBase
from torchvision import transforms as tt

class ViTB16(ClassifierBase):
    def __init__(self, path_ckpt: str):
        super().__init__()

        wgts = ViT_B_16_Weights.IMAGENET1K_V1
        self.id_to_cls = wgts.meta['categories']
        self.model = vit_b_16(weights = wgts)
        self.transforms = wgts.transforms()
        self.eval()

    def use_functional_relu_only(self):
        pass

    def load_ckpt(self, path_ckpt):
        pass

    def forward(self, x):
        # NOTE: Input is required to be in [0, 1] range
        x = self.transforms(x)
        x = self.model(x)
        return x

    def pred_prob(self, x):
        x = self(x)
        return F.softmax(x, dim = 1)
    
    def pred_label(self, x):
        x = self.pred_prob(x)
        return x.argmax(dim = 1).long()