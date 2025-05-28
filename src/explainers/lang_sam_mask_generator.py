import torch
from torchvision.transforms.functional import to_pil_image
from torch.nn.functional import softmax

from .base import AttributionMethod

import logging
log = logging.getLogger(__name__)


class LangSAMMaskGenerator(AttributionMethod):


    def __init__(self, *args, **kwargs):
        super(LangSAMMaskGenerator, self).__init__(*args, **kwargs)
        self.use_binary_masks = self.get_attr_maps_kwargs.pop("use_binary_masks")


    @torch.no_grad()
    def get_attr_maps(self, x, target):
        masks_stack = torch.zeros_like(x)

        for img_idx, img in enumerate(x):
            # convert to pil (required by lang_sam)
            img_pil = to_pil_image(img)

            # obtain masks predictions
            masks, logit_masks, boxes, phrases, logits = self.method.predict(img_pil, **self.get_attr_maps_kwargs)

            if not self.use_binary_masks:
                # zero out final probabilities where masks are already zeroed out
                logit_masks[masks == 0] = -torch.inf

                # convert all masks to a single binary mask
                n, h, w = logit_masks.shape
                prob_mask = softmax(logit_masks.view(n, -1), dim=1).sum(dim=0)
                mask = (prob_mask - prob_mask.min()) / (prob_mask.max() - prob_mask.min())
                mask = mask.view(h, w)

            else:
                # binary masks will ignore area constraints
                mask = (masks.sum(0) > 0).int()

            # repeat channel 3 times and save
            log.info(f"Obtained mask for text_prompt={self.get_attr_maps_kwargs['text_prompt']} with area={get_mask_area(mask)}")
            masks_stack[img_idx] = mask.unsqueeze(0).repeat(3, 1, 1)

        return masks_stack
    

def get_mask_area(mask):
    return (mask != 0).sum() / mask.numel()