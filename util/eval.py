import torch
from module.metric import MetricBase


class EvaluatePack(object):
    def __init__(self,
                 metrics,
                 loss=None,
                 use_gpu=True
                 ):
        """

        Args:
            metrics: a dict of name to metric object
            loss:
        """
        self.metrics = metrics
        for name, m in metrics.items():
            if not isinstance(m, MetricBase):
                raise ValueError('The metirc must be subclass of MetricBase, but it is {}'.format(m))
        self.loss = loss

        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')

    def compute_metric(self, data_loader):
        result = {}
        # compute metrics per batch
        for idx, (inputs, targets) in enumerate(data_loader):
            result = self.compute_metric_step(inputs, targets)
        # return the final result
        for name, metirc in self.metrics.items():
            result[name] = metirc.streaming_value()

        return result

    def compute_metric_step(self, inputs, targets):
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        ret = {}
        for name, metric in self.metrics.items():
            ret[name] = metric(inputs, targets)
        return ret
