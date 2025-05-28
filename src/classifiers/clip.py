from .base import ClassifierBase

import clip
import json
import logging
import numpy as np
import torch

from functools import cached_property
from torchvision import transforms


log = logging.getLogger(__name__)


class ShiftTransform(torch.nn.Module):
    """ ShiftTransform between two formats:
        Possible formats: 
         - "01" -  Inputs are in range [0, 1]
         - "-11" - Input are in range [-1, 1]
    """
    def __init__(self, input_format: str, output_format: str):
        super().__init__()
        self.input_format = input_format
        self.output_format = output_format

        if input_format not in ['01', '-11']:
            log.warning(f"Input format {input_format} not supported.")
            log.warning("No transformation will be applied.")

    def forward(self, x):
        if self.input_format == '01' and self.output_format == '-11':
            return 2 * x - 1
        elif self.input_format == '-11' and self.output_format == '01':
            return (x + 1) / 2
        else:
            return x


def get_preprocess_function(model_name: str):
    if model_name == 'ViT-B/32':
        return transforms.Compose([
            transforms.CenterCrop(224),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        ])
    else:
        raise ValueError(f"Model {model_name} not supported.")
        


class ClipZeroShot(ClassifierBase):
    def __init__(self, path_ckpt: str, model_name: str, labels_path: str):
        super().__init__()

        self.model_name = model_name
        self.labels_path = labels_path

        self.model, _ = clip.load(self.model_name, device = 'cpu')
        self.transforms = get_preprocess_function(self.model_name)

        self.model.eval()

        # dummy parameter to get device at any moment
        self.device_param = torch.nn.Parameter(torch.empty(0))


    @cached_property
    def labels(self):
        with open(self.labels_path, 'r') as f:
            label_list = json.load(f) # This should be a list
        assert isinstance(label_list, list)
        return label_list
        

    @cached_property
    def label_embeddings(self):
        labels = self.labels
        with torch.no_grad():
            text_inputs = torch.cat([
                clip.tokenize(f"a photo of a {label}") for label in labels
            ]).to(self.device_param.device)

            image_features = self.model.encode_text(text_inputs).float()
            image_features /= image_features.norm(dim = -1, keepdim = True)

        return image_features


    def forward(self, x):
        # x should be of shape [N, 3, 256, 256]
        # Input is required to be in [0, 1] range

        text_features = self.label_embeddings

        x = self.transforms(x)
        image_features = self.model.encode_image(x).float()
        norm = image_features.norm(dim = -1, keepdim = True).detach()
        image_features = image_features / norm

        similarity = (100.0 * image_features @ text_features.T)
        return similarity

    def load_ckpt(self, path_ckpt):
        pass

    def use_functional_relu_only(self):
        pass

    def pred_prob(self, x):
        return self.forward(x).softmax(dim = -1)

    def pred_label(self, x):
        logits = self.forward(x)
        return logits.argmax(dim = -1).long()



    