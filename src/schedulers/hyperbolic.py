from .base import SchedulerBase

class HyperbolicScheduler(SchedulerBase):
    start: float
    end: float
    gamma: float

    def __init__(self, total_timesteps: int, start: float, end: float = 0, gamma: float = 1, *args, **kwargs):
        super().__init__(total_timesteps, *args, **kwargs)
        self.start = start
        self.end = end
        self.gamma = gamma

    def at(self, timestep: int) -> float:
        alfa = 1 - (timestep / self.total_timesteps)
        a = (self.start - self.end) * (1 + self.gamma) / self.gamma
        b = self.start - a
        return a / (1 + self.gamma * alfa) + b