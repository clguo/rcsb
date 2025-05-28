from abc import ABC, abstractmethod
from torch import nn
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import wandb

class SchedulerBase(ABC, nn.Module):
    total_timesteps: int

    def __init__(self, total_timesteps: int, *args, **kwargs):
        self.total_timesteps = total_timesteps
        super().__init__(*args, **kwargs)
    
    @abstractmethod 
    def at(self, timestep: int, *args, **kwargs) -> float:
        """Get schedule value at a given timestep according to the scheduler implementation"""
        pass

    def __call__(self, timestep: int) -> float:
        return self.at(timestep)
        
    def log_scheduler_plot(self, name: str):
        fig, ax = plt.subplots()
        timesteps = np.arange(self.total_timesteps)
        values = np.array([self.at(timestep) for timestep in timesteps])
        ax.plot(timesteps, values)
        max_tick = self.total_timesteps + 1
        ticks = np.arange(0, max_tick, max_tick // 10)
        ax.set_xticks(ticks)
        ax.invert_xaxis()
        sns.set_theme()
        sns.set_style("whitegrid")
        wandb.log(
            {f"scheduler/{name}": wandb.Plotly.make_plot_media(fig)}
        )
        plt.close(fig)




    
