from .base import SchedulerBase
from utils.misc import get_timestep_value
from scipy.stats import norm
from functools import partial

class GaussianScheduler(SchedulerBase):
    max_value: float
    min_value: float
    max_at: int
    alfa_max: float
    scale: float

    def __init__(self, total_timesteps: int, max_value: float, min_value: float = 0, max_at: int | str = "50p", scale = 1, * args, **kwargs):
        super().__init__(total_timesteps, *args, **kwargs)
        self.max_value = max_value
        self.min_value = min_value
        self.max_at = get_timestep_value(max_at, total_timesteps)
        self.scale = scale

        self.alfa_max = 1 - (self.max_at / self.total_timesteps)

        density = partial(norm.pdf, loc = self.alfa_max, scale = self.scale)
        f_min = min(density(0), density(1))
        f_max = density(self.alfa_max)
        assert f_max > f_min
        self.gauss01 = lambda alfa: (density(alfa) - f_min) / (f_max - f_min)

    def at(self, timestep: int) -> float:
        alfa = 1 - (timestep / self.total_timesteps)
        return (self.max_value - self.min_value) * self.gauss01(alfa) + self.min_value