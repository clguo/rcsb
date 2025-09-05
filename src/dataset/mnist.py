import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from pathlib import Path
import sys
from functools import cache

import logging
log = logging.getLogger(__name__)


class MNISTDataset(Dataset):
    def __init__(
            self, 
            name: str,
            split: str,
            path_data: str, 
            path_labels: str, 
            path_predictions: str, 
            n_samples: int, 
            filter_id: int,
            n_skip: int):
        super().__init__()
        assert split in ['train', 'validation']
        self.name = name
        self.path_data = Path(path_data)
        self.split = split
        self.img_size = 32
        self.n_samples = n_samples
        self.filter_id = filter_id
        self.n_skip = n_skip

        if len(list(self.path_data.glob("*"))) == 0:
            self.prepare_data()
        
        train = (split == 'train')
        self.mnist_dataset = MNIST(
            self.path_data,
            train = train,
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
            ])
        )

        self.length = min(len(self.mnist_dataset), n_samples)

        self.predictions = pd.read_csv(path_predictions, index_col = "idx") if path_predictions else None
        self.filter_data(filter_id, n_skip)

    def prepare_data(self):
        MNIST(self.path_data, train=True, download=True)
        MNIST(self.path_data, train=False, download=True)

    def __len__(self):
        return self.length

    @cache
    def get_labels(self):
        return [self.mnist_dataset[i][1] for i in range(len(self.mnist_dataset))]

    
    def __getitem__(self, index):
        index = self.map_index(index)
        img, label = self.mnist_dataset[index] 

        if isinstance(self.predictions, pd.DataFrame):
            pred_label = self.predictions.loc[index]
        else:
            pred_label = -1

        return img, index, label,  pred_label.item()


    def filter_data(self, filter_id, n_skip):

        if filter_id is not None:
            gt_labels = pd.DataFrame(
                self.get_labels(),
                columns=['gt_label'],
            )

            assert len(gt_labels) == len(self.predictions), \
                "Length of gt_labels and predictions should be same. Probably a wrong predictions file is set."

            if self.predictions is None or (
                self.predictions[gt_labels.gt_label == filter_id]["pred_label"].isna().any()
                ):
                log.error(f'Filtering is not possible without predictions for label: {filter_id}')
                log.error(f'Please compute predictions for gt_label: {filter_id}')
                sys.exit(1)

            joint = self.predictions.join(gt_labels)
            # filter indices where filter_id is the gt label and the prediction is correct
            joint_filtered = joint[(joint.pred_label == filter_id) & (joint.pred_label == joint.gt_label)]
            filtered_idx = joint_filtered.sort_index().index.values

            # create index mapping
            self.map_index = lambda x: int(filtered_idx[x + n_skip])
            # change length
            self.length = min(len(filtered_idx), self.n_samples)
        else:
            # if no filtering, index is mapped with indentity
            self.map_index = lambda x: x + n_skip


    def get_guidance_classes(self, config, fabric, batch_labels):
        
        n_inp = config.exp.n_inpaints
        guidance_classes = []

        for label in batch_labels:
            if config.exp.guide_id is not None:
                guidance_classes.append([config.exp.guide_id] * n_inp)

        guidance_classes = torch.tensor(sum(guidance_classes, []))
        return guidance_classes