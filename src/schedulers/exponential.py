from .base import SchedulerBase
from numpy import exp, log

class ExponentialScheduler(SchedulerBase):
    start: float
    gamma : float
    end: float

    def __init__(self, total_timesteps: int, start: float, end: float = 0, gamma: float = 1, *args, **kwargs):
        super().__init__(total_timesteps, *args, **kwargs)
        self.start = start
        self.end = end
        self.gamma = gamma

    def at(self, timestep: int) -> float:
        alfa = 1 - (timestep / self.total_timesteps)
        a = (self.end - self.start) / (exp(-self.gamma) - 1)
        b = self.start - a
        return a * exp(-self.gamma * alfa) + b