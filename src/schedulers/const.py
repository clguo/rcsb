from .base import SchedulerBase

class ConstScheduler(SchedulerBase):
    value: float

    def __init__(self, total_timesteps: int, value: float, *args, **kwargs):
        super().__init__(total_timesteps, *args, **kwargs)
        self.value = value

    def at(self, timestep: int) -> float:
        return self.value


    