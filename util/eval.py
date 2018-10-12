import torch


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
        self.loss = loss

        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')

    def compute_metric(self, data_loader):
        ret = {}
        for idx, (inputs, targets) in enumerate(data_loader):
            result = self.compute_metric_step(inputs, targets)
            ret = result
            if idx == 0:
                continue

            ret = self.streaming_update(src_result=ret, next_result=result)
        return ret

    def compute_metric_step(self, inputs, targets):
        inputs.to(self.device)
        targets.to(self.device)
        ret = {}
        for name, metric in self.metrics.items():
            ret[name] = metric(inputs, targets)
        return ret

    def streaming_update(self, src_result, next_result):
        """

        Args:
            src_result: a dict
            next_result:  a dict

        Returns:

        """
        new_result = {}
        for name, value in src_result.items():
            metric_obj = self.metrics[name]

            new_value = metric_obj.update(value, next_result[name])

            new_result[name] = new_value

        return new_result
