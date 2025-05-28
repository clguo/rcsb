import torch
from omegaconf import DictConfig
from hydra.utils import instantiate
from .compute_clf_preds import get_output_path
from tqdm import tqdm
from pathlib import Path

import utils

from dataset import CustomMaskImagenetDataset

import logging

log = logging.getLogger(__name__)

torch.set_float32_matmul_precision = "high"


def get_fabric(config):
    fabric = instantiate(config.fabric)
    fabric.seed_everything(config.exp.seed)
    fabric.launch()
    return fabric


def get_components(config, fabric):
    # instantiate models
    classifier = instantiate(config.classifier)
    explainer = instantiate(config.explainer)(model=classifier)
    evaluator = instantiate(config.evaluation)(model=classifier)

    if config.guidance.uses_target_clf:
        guidance = fabric.setup(instantiate(config.guidance)(model=classifier))
    else:
        guidance = fabric.setup(instantiate(config.guidance))

    inpainter = instantiate(config.inpainter)(guidance=guidance)

    # compile modules with torch
    inpainter = torch.compile(inpainter)

    # setup modules with fabric
    explainer = fabric.setup(explainer)
    evaluator = fabric.setup(evaluator)
    inpainter = fabric.setup(inpainter)

    return explainer, inpainter, evaluator


def get_dataloader(config, fabric):
    # Set path_predictions on the run based on the classifier and dataset
    if config.dataset.dataset.path_predictions is None:
        log.info("Auto-Setting path_predictions")
        path_output = get_output_path(config)
        predictions_path: Path = path_output / "data.csv"
        log.info(f"path_predictions set to: {predictions_path}")
        assert predictions_path.exists(), (
            f"Predictions file not found at {predictions_path}"
        )
        config.dataset.dataset.path_predictions = predictions_path.absolute()

    return fabric.setup_dataloaders(instantiate(config.dataset))


def run(config: DictConfig):
    utils.preprocess_config(config)
    utils.setup_wandb(config)

    log.info("Launching Fabric")
    fabric = get_fabric(config)
    batch_multip = config.exp.batch_multip

    log.info("Building components")
    explainer, inpainter, evaluator = get_components(config, fabric)

    log.info("Initializing dataloader")
    dataloader = get_dataloader(config, fabric)

    use_custom_masks = isinstance(dataloader.dataset, CustomMaskImagenetDataset)

    with fabric.init_tensor():
        print(len(dataloader))
        for idx, batch in tqdm(
            enumerate(dataloader), total=len(dataloader), desc="Batches"
        ):
            log.info(f"Batch: {idx}")
            if not use_custom_masks:
                batch_imgs, batch_idx, batch_labels, batch_pred_labels = batch
            else:
                batch_imgs, batch_idx, batch_labels, batch_pred_labels, batch_masks = (
                    batch
                )
            utils.log_imgs(batch_idx, batch_imgs, "images")

            ## 1. get attribution maps
            log.info("Computing attribution maps")
            target = utils.get_target_id(config, batch_pred_labels)
            target = target.long()  # Make sure targets are ints
            if not isinstance(dataloader.dataset, CustomMaskImagenetDataset):
                batch_maps = explainer.get_attr_maps(batch_imgs, target=target)
            else:
                batch_maps = batch_masks
            utils.log_attr_maps(batch_idx, batch_maps)

            ## 2. postprocessing
            log.info("Postprocessing attribution maps")
            if not isinstance(dataloader.dataset, CustomMaskImagenetDataset):
                batch_maps_post = explainer.postprocess_attr_maps(batch_maps)
            else:
                batch_maps_post = batch_maps
            utils.log_imgs(batch_idx, batch_maps_post.unsqueeze(1), "attr_maps_post")
            utils.log_img_mask_overlay(batch_maps_post, batch_imgs)

            ## 3. inpainting
            log.info("Beginning inpainting")
            batch_inps = [None] * batch_multip
            batch_guidance_classes = [None] * batch_multip
            batch_imgs_rep, batch_maps_rep = utils.get_batch_to_inp(
                config, batch_imgs, batch_maps_post
            )

            for idx_inner in range(batch_multip):
                log.info(f"Inpainting batch: {idx_inner + 1}/{batch_multip}")
                batch_guidance_classes_ = dataloader.dataset.get_guidance_classes(
                    config, fabric, batch_labels, batch_pred_labels
                )
                batch_inps_ = inpainter.inpaint(
                    batch_imgs_rep, batch_maps_rep, batch_guidance_classes_
                )
                batch_inps[idx_inner] = batch_inps_
                batch_guidance_classes[idx_inner] = batch_guidance_classes_

            batch_inps = torch.cat(batch_inps)
            batch_guidance_classes = torch.cat(batch_guidance_classes)
            utils.log_imgs(batch_idx, batch_inps, "inpaints")

            ## 4. evaluate
            log.info("Evaluating inpaints")
            eval_input = {
                "batch_idx": batch_idx,
                "batch_imgs": batch_imgs,
                "batch_inps": batch_inps,
                "batch_labels": batch_labels,
                "batch_pred_labels": batch_pred_labels,
                "batch_guidance_classes": batch_guidance_classes,
            }
            evaluator.evaluate(config, eval_input)
