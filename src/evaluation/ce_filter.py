import torch
import wandb
import pandas as pd
from torchvision.utils import make_grid

from .base import Metric

class CEFilter(Metric):

    def __init__(self):
        super().__init__()

    def compute(self, config, classifier, inputs):
        assert {'batch_idx', 
                'batch_inps', 
                'batch_pred_labels'} <= inputs.keys()

        batch_idx = inputs['batch_idx']
        batch_inps = inputs['batch_inps']
        batch_guidance_classes = inputs['batch_guidance_classes']
        batch_pred_labels_orig = inputs['batch_pred_labels']

        n_splits = batch_idx.shape[0]
        n_imgs_per_split = batch_inps.shape[0] // n_splits
    
        with torch.no_grad():
            batch_pred_labels_inps = classifier.pred_label(batch_inps)

        batch_inps_s = batch_inps.split(n_imgs_per_split)
        batch_pred_labels_inps_s = batch_pred_labels_inps.split(n_imgs_per_split)
        batch_guidance_classes = batch_guidance_classes.split(n_imgs_per_split)

        iterator = zip(batch_pred_labels_orig, batch_inps_s, batch_pred_labels_inps_s, batch_guidance_classes)
        guide_id = config.exp.guide_id
        
        for label, batch_inps_, batch_pred_labels_, batch_guidance_classes_ in iterator:

            if config.exp.task == "multiclass":
                # if guide_id is provided, filter images which flip 
                # the decision to this label, else simply check if decision is flipped
                if guide_id is not None:
                    ce_idx = batch_pred_labels_ == guide_id
                else:
                    ce_idx = batch_pred_labels_ != label

            elif config.exp.task == "multilabel":
                ce_idx = batch_pred_labels_ == batch_guidance_classes_

            else:
                raise NotImplementedError

            batch_inps_[~ce_idx] = 0.
            grid = make_grid(batch_inps_)
            wandb.log({'counterfactuals': wandb.Image(grid)})