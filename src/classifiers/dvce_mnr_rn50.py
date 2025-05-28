import torch
import torch.nn.functional as F
from .base import ClassifierBase
from torchvision.models import resnet50
from torchvision import transforms as tt

class DVCEMNRResNet50(ClassifierBase):
    def __init__(self, path_ckpt: str):
        super().__init__()
        self.model = resnet50(pretrained = False)
        self.transforms = tt.Compose([
            tt.CenterCrop(224),
            tt.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ])
        self.load_ckpt(path_ckpt)
        self.eval()

    def use_functional_relu_only(self):
        pass

    def load_ckpt(self, path_ckpt):
        state_dict = torch.load(path_ckpt)
        self.model.load_state_dict(state_dict)

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