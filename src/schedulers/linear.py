from .base import SchedulerBase

class LinearScheduler(SchedulerBase):
    start: float
    end: float

    def __init__(self, total_timesteps: int, start: float, end: float = 0, *args, **kwargs):
        super().__init__(total_timesteps, *args, **kwargs)
        self.start = start
        self.end = end

    def at(self, timestep: int) -> float:
        alfa = 1 - (timestep / self.total_timesteps)
        return ((1 - alfa) * self.start) + (alfa * self.end)