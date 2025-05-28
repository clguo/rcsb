import PIL
import torch
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .base import Metric

wandb.Table.MAX_ARTIFACT_ROWS = 200_000_000
wandb.Table.MAX_ROWS = 200_000_000

class Predictions(Metric):

    def __init__(self, task: str, n_classes: int):
        super().__init__()
        self.task = task
        self.n_classes = n_classes
        self.table = self.get_empty_table()

    def get_empty_table(self):
        self.columns = ['index', 'is_cf', 'pred_id', 'guide_id'] + \
            [f'prob_orig_{i}' for i in range(self.n_classes)] + \
            [f'prob_inp_{i}' for i in range(self.n_classes)]
        df = pd.DataFrame(columns = self.columns)
        table = wandb.Table(dataframe = df, allow_mixed_types = True)
        return table

    def forward(self, x):
        pass

    def log_results(self, batch_idx, pred_prob_imgs, pred_prob_inps, batch_guide_idx):
        # extract number of inpaints per unique image
        n_inps = pred_prob_inps.shape[0] // pred_prob_imgs.shape[0]

        # log image indices
        for val in batch_idx.tolist():
            wandb.log({'misc/img_idx': val})

        # duplicate data using n_inps
        batch_idx = batch_idx.repeat_interleave(n_inps, 0)
        pred_prob_imgs = pred_prob_imgs.repeat_interleave(n_inps, 0)

        target_imgs = pred_prob_imgs.argmax(dim = 1)
        target_inps = pred_prob_inps.argmax(dim = 1)
        is_cf = (target_imgs != target_inps).int()

        data = torch.hstack([
            batch_idx.unsqueeze(1), 
            is_cf.unsqueeze(1), 
            target_inps.unsqueeze(1), 
            batch_guide_idx.unsqueeze(1),
            pred_prob_imgs, 
            pred_prob_inps]).numpy(force = True)
        
        
        df = pd.DataFrame(data = data, columns = self.columns)

        # log flip rate separately for each image
        fr_per_img = df.groupby('index')['is_cf'].mean().values
        for val in fr_per_img:
            wandb.log({'predictions/flip_rate': val})

        # log guided flip rate separately for each image
        fr_g_per_img = df.groupby('index').apply(lambda x: (x[x.is_cf == 1.].pred_id == x[x.is_cf == 1.].guide_id).mean())
        for val in fr_g_per_img.values:
            if np.isnan(val):
                val = 0.
            wandb.log({'predictions/flip_rate_guided': val})

        # log change in prediction for initially predicted label
        prob_orig_cols = [e for e in df.columns if e.startswith('prob_orig')]
        prob_inps_cols = [e for e in df.columns if e.startswith('prob_inp')]

        def _max_val_in_group(group):
            return group.max(axis = 1)
        
        def _max_col_in_group(group):
            return group.idxmax(axis = 1)

        def _get_label_from_col_name(name):
            return int(name.split('_')[-1])

        # find prob predicted for each initial image
        pred_prob_orig = df.groupby('index').first().\
            groupby('index')[prob_orig_cols].apply(_max_val_in_group).values.flatten()
        
        # find name of the column corresponding to the predicted label
        pred_prob_orig_name = df.groupby('index').first().\
            groupby('index')[prob_orig_cols].apply(_max_col_in_group).values.flatten()
        
        # extract predicted label id
        pred_label_id = [_get_label_from_col_name(e) for e in pred_prob_orig_name]
        # extract name of the predicted label in inps columns
        pred_label_inps_name = [prob_inps_cols[idx] for idx in pred_label_id]

        iterator = zip(pred_label_inps_name, pred_prob_orig, df.groupby('index'))

        def log_pred_label_count_plot(group_df):
            import tempfile
            fig, ax = plt.subplots()
            tmp_df = group_df.pred_id.value_counts().sort_index()
            ax = tmp_df.plot(kind="barh")
            fig.tight_layout()
            with tempfile.TemporaryFile() as fp:
                plt.savefig(fp)
                img = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
                wandb.log({"predictions/predicted_labels": wandb.Image(img)})
            plt.close(fig)

        # iterate over groups
        for pred_label_inp_name, pred_prob_val, (_, group) in iterator:
            # log mean probability delta and initial probability
            deltas = group[pred_label_inp_name] - pred_prob_val
            mean_delta = deltas.mean()
            wandb.log({
                'predictions/mean_prob_delta': mean_delta,
                'predictions/initial_prob': pred_prob_val})
            log_pred_label_count_plot(group)
        

    def compute(self, config, classifier, inputs):
        assert {'batch_guidance_classes', 'batch_idx', 'batch_imgs', 'batch_inps'} \
            <= inputs.keys()

        batch_idx = inputs['batch_idx']
        batch_imgs = inputs['batch_imgs']
        batch_inps = inputs['batch_inps']
        batch_guidance_classes = inputs['batch_guidance_classes']

        with torch.no_grad():
            pred_prob_imgs = classifier.pred_prob(batch_imgs)
            pred_prob_inps = classifier.pred_prob(batch_inps)

        self.log_results(
            batch_idx, pred_prob_imgs, pred_prob_inps, batch_guidance_classes)