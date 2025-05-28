from .base import SchedulerBase
from dataclasses import dataclass
from utils.misc import get_timestep_value

@dataclass
class Interval:
    start: float
    end: float

    def __post_init__(self):
        if self.start >= self.end:
            self.start, self.end = self.end, self.start
    
    def __contains__(self, value: float) -> bool:
        return self.start <= value <= self.end


class IntervalScheduler(SchedulerBase):
    value: float
    start_timestep: int
    end_timestep: int

    def __init__(self, total_timesteps: int, value: float, start_timestep: int | str = "100p", end_timestep: int | str = "0p", min_value = 0, *args, **kwargs):
        super().__init__(total_timesteps, *args, **kwargs)
        self.value = value
        self.min_value = min_value
        self.start_timestep = get_timestep_value(start_timestep, total_timesteps)
        self.end_timestep = get_timestep_value(end_timestep, total_timesteps)

        self.interval = Interval(self.start_timestep, self.end_timestep)

    def at(self, timestep: int) -> float:
        if timestep in self.interval:
            return self.value
        else:
            return self.min_value