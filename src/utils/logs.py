import torch
import wandb
from torchvision.utils import make_grid, draw_segmentation_masks
from torchvision.transforms.functional import convert_image_dtype
from explainers.utils import normalize_attr

def log_attr_maps(batch_idx, batch_attrs):
    n_splits = batch_idx.shape[0]
    n_imgs_per_split = batch_attrs.shape[0] // n_splits

    for batch_attrs_ in batch_attrs.split(n_imgs_per_split):
        batch_attrs_pos = normalize_attr(batch_attrs_, 'positive').unsqueeze(1)
        batch_attrs_neg = normalize_attr(batch_attrs_, 'negative').unsqueeze(1)
        batch_attrs_abs = normalize_attr(batch_attrs_, 'absolute_value').unsqueeze(1)
        grid_pos = make_grid(batch_attrs_pos, normalize = True, scale_each = True)
        grid_neg = make_grid(batch_attrs_neg, normalize = True, scale_each = True)
        grid_abs = make_grid(batch_attrs_abs, normalize = True, scale_each = True)
        wandb.log({'attr_maps/positive': wandb.Image(grid_pos), 
                    'attr_maps/negative': wandb.Image(grid_neg),
                    'attr_maps/absolute': wandb.Image(grid_abs)})

def log_imgs(batch_idx, batch_imgs, tag):
    n_splits = batch_idx.shape[0]
    n_imgs_per_split = batch_imgs.shape[0] // n_splits
    
    for batch_imgs_ in batch_imgs.split(n_imgs_per_split):
        grid = make_grid(batch_imgs_.float())
        wandb.log({tag: wandb.Image(grid)})

def log_img_mask_overlay(batch_maps_post, batch_imgs):
    for img, attr_map in zip(batch_imgs, batch_maps_post):
        if img.shape[0] == 1: # Convert to RGB if Grayscale
            img = img.repeat(3, 1, 1)
        overlayed = draw_segmentation_masks(
            convert_image_dtype(img, torch.uint8), 
            masks=attr_map.bool(),
            colors="yellow", alpha=0.4)
        wandb.log({"attr_maps_post_overlayed": wandb.Image(overlayed.float())})
