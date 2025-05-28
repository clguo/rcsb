import torch
import wandb
from pathlib import Path
import omegaconf
from omegaconf import DictConfig
import os

def extract_output_dir(config: DictConfig) -> Path:
    '''
    Extracts path to output directory created by Hydra as pathlib.Path instance
    '''
    date = '/'.join(list(config._metadata.resolver_cache['now'].values()))
    output_dir = Path.cwd() / 'outputs' / date
    return output_dir

def preprocess_config(config):
    config.exp.log_dir = extract_output_dir(config)

def setup_wandb(config):
    group, name = str(config.exp.log_dir).split('/')[-2:]
    wandb_config = omegaconf.OmegaConf.to_container(
        config, resolve = True, throw_on_missing = True
    )
    name = os.getenv("WANDB_RUN_NAME", name)
    
    if "id" in config.wandb.keys() and config.wandb.id is not None:
        run_id = config.wandb.id
    else:
        run_id = None

    return wandb.init(
        project = config.wandb.project,
        entity = config.wandb.entity,
        dir = config.exp.log_dir,
        group = group,
        name = name,
        config = wandb_config,
        sync_tensorboard = True,
        tags = config.wandb.tags,
        id = run_id
    )
            

def get_batch_to_inp(config, batch_imgs, batch_maps):
    '''
    Repeats each image and map as indicated by the
    config.exp.n_inpaints argument.
    '''
    n_inp = config.exp.n_inpaints
    batch_imgs_rep = batch_imgs.repeat_interleave(n_inp, 0)
    batch_maps_rep = batch_maps.repeat_interleave(n_inp, 0)
    return batch_imgs_rep, batch_maps_rep


def get_target_id(config, batch_pred_labels):
    '''
    target_id indicates class for which attribution map is generated
    for multilabel, this is determined by the user
    for multiclass, this is determined by the classifier's prediction
        if config.exp.target_id is None, else it is equal to it
    '''
    if config.exp.task == 'multiclass':

        if config.exp.target_id is not None:
            target = torch.ones_like(batch_pred_labels) * config.exp.target_id

        else:
            target = batch_pred_labels

    elif config.exp.task == 'multilabel':
        # here we assume that a multilabel classifier was converted to a
        # model with (p, 1 - p) output pair, where p is the probability of
        # the positive target class. therefore, no matter which target label was
        # chosen, we always compute the attribution for p.
        target = torch.zeros_like(batch_pred_labels)

    else:
        raise NotImplementedError('Task not yet implemented')

    return target


def unlist(item):
    if hasattr(item, '__len__') and hasattr(item, '__getitem__'):
        # we always have the same t across batch
        if isinstance(item, torch.Tensor):
            return item.flatten()[0].item()
        assert len(item) == 1, "Item is not a singleton"
        return unlist(item[0])
    return item

def get_timestep_value(relative_value: int | str, max_timestep):
    if isinstance(relative_value, str):
        assert relative_value[-1] == 'p', 'Relative value must be a percentage'
        relative_value = float(relative_value[:-1]) / 100
        relavtive_value_to_process = 1 - relative_value
        return int(max_timestep * relavtive_value_to_process)
    return int(relative_value)
