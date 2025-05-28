from torch import nn

class Evaluator(nn.Module):

    def __init__(self, model: nn.Module, metrics: nn.ModuleList):
        super().__init__()

        self.model = model
        self.metrics = metrics

    def evaluate(self, config, inputs):
        for metric in self.metrics:
            metric.compute(config, self.model, inputs)