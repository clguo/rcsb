import torch
from pathlib import Path
from tqdm import tqdm
from omegaconf import DictConfig
from hydra.utils import instantiate
import sys

import logging
log = logging.getLogger(__name__)

def get_fabric(config):
    fabric = instantiate(config.fabric)
    fabric.seed_everything(config.exp.seed)
    fabric.launch()
    return fabric

def get_components(config, fabric):
    classifier = fabric.setup(instantiate(config.classifier))
    return classifier

def get_dataloader(config, fabric, path_predictions):
    return fabric.setup_dataloaders(instantiate(
        config.dataset, dataset = {
            "path_predictions":path_predictions
        }
    ))

def get_output_path(config):
    path_meta = Path('data/metadata')
    dataset_name = config.dataset.dataset.name
    split_name = config.dataset.dataset.split
    assert dataset_name is not None
    assert split_name is not None
    clf_name = config.classifier._target_
    path_output = path_meta / dataset_name / 'predictions' / split_name / clf_name
    path_output.mkdir(parents = True, exist_ok = True)
    return path_output


def run(config : DictConfig):
    log.info(f'Launching Fabric')
    fabric = get_fabric(config)

    log.info(f'Building components')
    classifier = get_components(config, fabric)

    log.info(f'Initializing dataloader')
    path_output = get_output_path(config)
    dataloader = get_dataloader(config, fabric, path_output / 'data.csv')
    dataset = dataloader.dataset

    assert hasattr(dataset, 'save_predictions')
    
    data = []

    if len(dataloader) == 0:
        log.info('No data to process')
        sys.exit(0)
    
    log.info("Number of samples to process: {}".format(len(dataloader.dataset)))
    log.info("Number of batches: {}".format(len(dataloader)))

    for idx, batch in tqdm(enumerate(dataloader), total = len(dataloader)):
        log.info(f'Batch: {idx}')
        batch_imgs, batch_idx, label, _ = batch

        log.info('Running inference')
        with torch.no_grad():
            batch_preds = classifier(batch_imgs)
            
        if config.exp.task == 'multiclass':
            batch_preds = batch_preds.argmax(dim = 1)
            batch_data = torch.stack([batch_idx, batch_preds]).T
            
        elif config.exp.task == "multilabel":
            batch_preds = (batch_preds > 0.).int()
            batch_data = torch.hstack([batch_idx.unsqueeze(1), batch_preds])

        data.append(batch_data)
    
    log.info('Gathering predictions from all processes')
    # in each process, concat batches
    data = torch.cat(data)
    
    # gather concatenated batches from each process
    # in the main process
    data = fabric.all_gather(data)

    if fabric.global_rank == 0:
        if fabric.world_size > 1:
            # concat across processes
            data = data.flatten(start_dim = 0, end_dim = 1)

        log.info('Saving predictions..')
        data = data.cpu()
        dataset.save_predictions(data)
        log.info('Predictions saved')

