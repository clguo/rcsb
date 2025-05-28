import torch
import torch.nn.functional as F
from .base import ClassifierBase
from torchvision.models import resnet50
from torchvision import transforms as tt

class MadryResNet50(ClassifierBase):
    def __init__(self, path_ckpt: str):
        super().__init__()

        self.model = resnet50(pretrained = False)
        self.transforms = tt.Compose([
            tt.CenterCrop(224),
            tt.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        ])
        self.load_ckpt(path_ckpt)
        self.eval()

    def use_functional_relu_only(self):
        pass

    def load_ckpt(self, path_ckpt):
        state_dict = torch.load(path_ckpt)["model"]
        model_state_dict = {
            key.removeprefix("module.model."): value
            for key, value in state_dict.items()
            if key.startswith("module.model")
        }
        self.model.load_state_dict(model_state_dict)

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