import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.io import read_image


import logging
log = logging.getLogger(__name__)


class CelebAHQDataset(Dataset):
    """
    torch.utils.data.Dataset for CelebA-HQ. Assumes that images are stored as:

    celebahq/
        imgs/
            <img_id>/
                img.png

    and that there are 30 000 images in total.
    """
    

    def __init__(
            self, 
            path_data: str, 
            path_labels: str, 
            path_predictions: str,
            name: str,
            split: str,
            label_id: int,
            n_samples: int,
            n_skip: int):
        
        super(CelebAHQDataset, self).__init__()

        self.paths = self.get_paths(path_data)
        self.labels = self.get_labels(path_labels)
        self.predictions = self.get_predictions(path_predictions)
        self.name = name
        self.split = split
        self.label_id = label_id
        self.length = min(len(self.paths), n_samples)
        self.map_index = self.get_map_index(n_skip)


    def get_paths(self, path):
        paths = [Path(path) / "imgs" / str(idx) / "img.png" for idx in range(30_000)]
        return paths
    

    def get_labels(self, path):
        return pd.read_csv(path, index_col = 0).drop("id", axis=1)
    

    def get_predictions(self, path):

        if not isinstance(path, Path):
            path = Path(path)

        if not path.exists():
            log.info("Creating dataset with no predictions")
            self.path_predictions = path
            return None
        else:
            return pd.read_csv(path, index_col=0)


    def get_map_index(self, n_skip):
        # decrease length by n_skip images
        self.length -= n_skip

        # return shifted identity
        return lambda x: x + n_skip


    def __len__(self):
        return self.length
    

    def get_guidance_classes(self, config, fabric, batch_labels, batch_pred_labels):
        # following the conversion of the classifier to a model outputing only (p, 1 - p)
        # pairs, where p represents the class of interest, we want to guide the model to the
        # opposite prediction. hence, if pred_label == 1, we increase 1 - p, otherwise p.
        return batch_pred_labels.repeat_interleave(config.exp.n_inpaints)
        
    
    def save_predictions(self, data: np.ndarray):
        data = pd.DataFrame(data[:, 1:], data[:, 0].numpy(force=True), self.labels.columns)
        data.to_csv(self.path_predictions, index=True)


    def __getitem__(self, index):
        index = self.map_index(index)
        path = self.paths[index]
        img = read_image(str(path)) / 255
        gt_label = -1 if self.label_id is None else self.labels.iloc[index, self.label_id]
        pred_label = -1 if self.predictions is None else self.predictions.iloc[index, self.label_id]
        log.info(f"Initial prediction: {pred_label.item()}")
        return img, index, gt_label, pred_label


