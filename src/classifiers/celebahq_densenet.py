import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as tt

from .base import ClassifierBase

import logging
log = logging.getLogger(__name__)


class Identity(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class DenseNet121(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.feat_extract = torchvision.models.densenet121(weights = None)
        self.feat_extract.classifier = Identity()
        self.output_size = 1024
    
    def forward(self, x):
        return self.feat_extract(x)


class DenseNet(ClassifierBase):
    """
    DenseNet architecture for CelebA-HQ. 
    """
    id_to_cls = ['male', 'smile', 'young']


    def __init__(self, path_ckpt: str, label_id: int):
        super().__init__()
        
        self.feat_extract = DenseNet121()
        self.classifier = torch.nn.Linear(self.feat_extract.output_size, 3)
        self.transforms = tt.Compose([
            tt.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))])
        self.label_id = label_id

        if label_id is None:
            log.info("Creating classifier with unspecified label_id")

        self.load_ckpt(path_ckpt)
        self.eval()


    def use_functional_relu_only(self):
        pass


    def load_ckpt(self, path_ckpt):
        ckpt = torch.load(path_ckpt, map_location = 'cpu')
        self.load_state_dict(ckpt["model_state_dict"])


    def forward(self, x):
        # NOTE: Input is required to be in [0, 1] range
        x = self.transforms(x)
        x = self.feat_extract(x)
        x = self.classifier(x)
        
        # every multilabel classifier can be converted to 'multiclass' classifier by:
        #   1. constraining the predictions to include only the label of interest
        #   2. modifying its output to provide pairs of (p, 1 - p), where p denotes
        #       the probability for the class of interest.

        if self.label_id is not None:
            
            # pick column correspondig to label of interest
            x = x[:, [self.label_id]]

            # add column with its repeated logits
            x = x.repeat(1, 2)

            # make it negative to later represent 1 - p, where p
            # is the probability determined by initial logits
            x[:, 1] = - x[:, 1]

        return x
    

    def pred_prob(self, x):
        x = self(x)
        return F.sigmoid(x)
    

    def pred_label(self, x):
        x = self.pred_prob(x)
        return x.argmax(dim = 1).long()