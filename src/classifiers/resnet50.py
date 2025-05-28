from torch import Tensor
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from .base import ClassifierBase

#https://github.com/pytorch/captum/issues/378
# in short for DeepLift we cannot use the same nn.Relu twice as in original implementation
from torchvision.models.resnet import BasicBlock, Bottleneck

def resnet_basic_block_forward_separate_relu(self,  x: Tensor):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = F.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    out += identity
    out = F.relu(out)

    return out

def resnet_bottleneck_forward_separate_relu(self, x: Tensor = None):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = F.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = F.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    out += identity
    out = F.relu(out)

    return out

def override_forward(module):
    if isinstance(module, BasicBlock):
        module.forward = resnet_basic_block_forward_separate_relu.__get__(module, BasicBlock)
    elif isinstance(module, Bottleneck):
        module.forward = resnet_bottleneck_forward_separate_relu.__get__(module, Bottleneck)
    for child in module.children():
        override_forward(child)

class ResNet50(ClassifierBase):
    def __init__(self, path_ckpt: str):
        super().__init__()

        wgts = ResNet50_Weights.IMAGENET1K_V1
        self.id_to_cls = wgts.meta['categories']
        self.model = resnet50(weights = wgts)
        self.transforms = wgts.transforms()
        self.eval()

    def use_functional_relu_only(self):
        """Converts nn.module.ReLU to F.relu in BasicBlock and Bottleneck submodules"""
        override_forward(self.model)

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