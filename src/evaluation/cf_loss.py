import torch
import wandb
import lpips
from pathlib import Path
import matplotlib.pyplot as plt

from .base import Metric

import logging
log = logging.getLogger(__name__)

class CFLoss(Metric):

    def __init__(
            self, 
            lpips: lpips.LPIPS,
            lambda_param: float,
            use_adaptive_lambda: bool):
        super().__init__()
        self.lpips = lpips
        self.lambda_param = lambda_param
        self.use_adaptive_lambda = use_adaptive_lambda

    def compute(self, config, classifier, inputs):
        assert {'batch_idx', 'batch_imgs', 'batch_inps'} <= inputs.keys()

        def scale_lpips_input(x):
            '''
            Scales x from [0, 1] to [-1, 1] range
            '''
            return (x - 0.5) * 2

        def log_plt_heatmap(x, name, n_cols = 8, max_1 = True):
            # for n_cols we follow the default parameter from make_grid to 
            # make each entry of the heatmap correspond to proper image
            n_cols = min(x.shape[0], n_cols)
            x = x.reshape(-1, n_cols).round(decimals = 3).numpy(force = True)
            fig, ax = plt.subplots()
            im = ax.imshow(x, vmin = 0., vmax = 1. if max_1 else None, cmap = 'autumn')
            for i in range(x.shape[0]):
                for j in range(n_cols):
                    ax.text(j, i, x[i, j], 
                        ha = 'center', va = 'center', color = '#000000')
            ax.set_yticks([])
            ax.set_xticks([])
            plt.colorbar(im)
            fig.tight_layout()
            wandb.log({f'cf_loss/{name}': wandb.Plotly.make_plot_media(fig)})
            plt.close(fig)

        batch_imgs = inputs['batch_imgs']
        batch_inps = inputs['batch_inps']
        n_inps = config.exp.n_inpaints * config.exp.batch_multip
        n_uniq = config.exp.n_unique
        batch_pred_labels = inputs['batch_pred_labels'].\
            repeat_interleave(n_inps).view(-1, 1)
        batch_pred_labels = batch_pred_labels.long()

        with torch.no_grad():
            batch_logits_all = classifier(batch_inps)
            batch_probs_all = classifier.pred_prob(batch_inps)
            batch_probs = batch_probs_all.gather(1, batch_pred_labels)

            if n_uniq > 1:
                batch_lpips = self.lpips(
                    scale_lpips_input(batch_imgs.repeat(config.exp.batch_multip, 1, 1, 1)), 
                    scale_lpips_input(batch_inps))

            else:
                batch_lpips = self.lpips(scale_lpips_input(batch_imgs), scale_lpips_input(batch_inps))

        batch_loss = batch_probs.view(-1) + self.lambda_param * batch_lpips.view(-1)

        log_plt_heatmap(batch_loss, 'loss', max_1 = False)
        log_plt_heatmap(batch_probs, 'probs')
        log_plt_heatmap(batch_lpips, 'lpips')
        
        batch_size = batch_probs_all.shape[0]
        n_inps_per_batch_per_uniq = config.exp.n_inpaints

        def log_plt_histogram(x):
            n_samples = len(x)
            fig = plt.figure()
            plt.hist(x, density=True, bins=max(int(n_samples ** (1/2)), 6))
            mean, var = x.mean().item(), x.var().item()
            q1, median, q3 = x.quantile(torch.Tensor([0.25, 0.5, 0.75])).tolist()
            fig.tight_layout()
            wandb.log({
                "cf_loss/density": wandb.Plotly.make_plot_media(fig),
                "cf_loss/mean": mean,
                "cf_loss/variance": var,
                "cf_loss/median": median,
                "cf_loss/q1": q1,
                "cf_loss/q3": q3,})
            plt.close(fig)

        def log_failed():
            fig = plt.figure()
            plt.plot([1., 1., 1., 1.])
            wandb.log({
                "cf_loss/density": wandb.Plotly.make_plot_media(fig),
                "cf_loss/mean": 0.0,
                "cf_loss/variance": 0.0,
                "cf_loss/median": 0.0,
                "cf_loss/q1": 0.0,
                "cf_loss/q3": 0.0,})
            plt.close(fig)

        for i, idx in enumerate(inputs["batch_idx"]):
            
            art = wandb.Artifact(
                name=f"prob_lpips_vals_idx-{idx.item()}",
                type="cf_loss_components")
            
            if n_uniq > 1:
                batch_probs_all_tmp = batch_probs_all.view(batch_size // n_uniq, n_uniq, -1)\
                    [:, i*n_inps_per_batch_per_uniq:((i + 1)*n_inps_per_batch_per_uniq)].reshape(batch_size // n_uniq, -1)
                batch_logits_all_tmp = batch_logits_all.view(batch_size // n_uniq, n_uniq, -1)\
                    [:, i*n_inps_per_batch_per_uniq:((i + 1)*n_inps_per_batch_per_uniq)].reshape(batch_size // n_uniq, -1)
                batch_lpips_tmp = batch_lpips.view(batch_size // n_uniq, n_uniq, -1)\
                    [:, i*n_inps_per_batch_per_uniq:((i + 1)*n_inps_per_batch_per_uniq)].reshape(batch_size // n_uniq, -1)
                
                art_data = {
                    "probs": batch_probs_all_tmp.cpu(),
                    "lpips": batch_lpips_tmp.cpu(),
                    "logits": batch_logits_all_tmp.cpu()}

                guide_idx = inputs["batch_guidance_classes"].view(batch_size // n_uniq, n_uniq, -1)\
                    [:, i*n_inps_per_batch_per_uniq:((i + 1)*n_inps_per_batch_per_uniq)].cpu()

            else:

                art_data = {
                    "probs": batch_probs_all.cpu(),
                    "lpips": batch_lpips.flatten(1).cpu(),
                    "logits": batch_logits_all.cpu()}

                guide_idx = inputs["batch_guidance_classes"].cpu()
            
            path_art = Path(wandb.run.dir) / "artifacts" /f"{art.name}.pt"
            path_art.parent.mkdir(parents=True, exist_ok=True)
            torch.save(art_data, path_art)
                
            art.add_file(path_art, is_tmp=True)
            wandb.log_artifact(art)

            cf_idx = art_data["probs"].argmax(dim=1) == guide_idx

            if any(cf_idx):
                try:
                    if self.use_adaptive_lambda:
                        var_probs = (art_data["probs"][cf_idx, guide_idx[cf_idx]]).var()
                        var_lpips = (1 - art_data["lpips"][cf_idx].flatten()).var()
                        lambda_param = var_lpips / (var_probs + var_lpips)
                    else:
                        lambda_param = self.lambda_param

                    hist_data = lambda_param * (art_data["probs"][cf_idx, guide_idx[cf_idx]]) + \
                        (1 - lambda_param) * (1 - art_data["lpips"][cf_idx].flatten())
                    log_plt_histogram(hist_data)
                except Exception as e:
                    log.error(f"Error in logging histogram: {e}")
                    log_failed()
            else:
                log.info("No counterfactuals found")
                log_failed()