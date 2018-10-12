import torch
from abc import abstractmethod
from sklearn.metrics import confusion_matrix


class MetricBase(object):
    def __init__(self, name=''):
        self._name = name

    @property
    def name(self):
        return self._name

    @abstractmethod
    def update(self, *args, **kwargs):
        return NotImplementedError

    @abstractmethod
    def forward(self, y_pred, y_true):
        return NotImplementedError

    @abstractmethod
    def streaming_value(self):
        return NotImplementedError

    def __call__(self, y_pred, y_true):
        with torch.no_grad():
            return self.forward(y_pred, y_true)


class Accuracy(MetricBase):
    def __init__(self, name='accuracy'):
        super(Accuracy, self).__init__(name)
        self.num_true = 0.
        self.total = 0.

    def update(self, num_true, total):
        self.num_true += num_true
        self.total += total

    def forward(self, y_pred, y_true):
        """

        Args:
            y_pred: 1-D tensor of shape [N, ]
            y_true: 1-D tensor of shape [N, ]

        Returns:

        """
        y_pred = y_pred.to(torch.float32)
        y_true = y_true.to(torch.float32)

        num_true = torch.sum(y_pred == y_true).float()

        total = float(y_true.size(0))

        self.update(num_true, total)

        return num_true / total

    def streaming_value(self):
        return self.num_true / self.total


class ConfusionMatrix(MetricBase):
    def __init__(self,
                 num_classes,
                 name='confusion_matrix'):
        """
        row -- gt
        col -- pred
        Args:
            num_classes: scalar
            name:
        """
        super(ConfusionMatrix, self).__init__(name)
        self._num_classes = num_classes
        self._total_cm = np.zeros([self._num_classes, self._num_classes], np.float32)

    def forward(self, y_pred, y_true):
        y_pred = y_pred.detach().numpy()
        y_true = y_true.detach().numpy()
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=list(range(self._num_classes)))
        self.update(cm)
        return cm

    def update(self, cm):
        self._total_cm += cm

    def streaming_value(self):
        return self._total_cm


class IoUPerClass(MetricBase):
    def __init__(self,
                 num_classes,
                 name='iou_per_class'):
        super(IoUPerClass, self).__init__(name)

        self.conf_mat = ConfusionMatrix(num_classes)

    def _forward(self, cm):
        """

        Args:
            cm: numpy array [num_classes, num_classes] row - gt, col - pred

        Returns:
            iou_per_class: float32 [num_classes, ]
        """
        sum_over_row = np.sum(cm, axis=0)
        sum_over_col = np.sum(cm, axis=1)
        diag = np.diag(cm)
        denominator = sum_over_row + sum_over_col - diag

        iou_per_class = diag / denominator

        return iou_per_class

    def forward(self, y_pred, y_true):
        cur_cm = self.conf_mat.forward(y_pred, y_true)
        self.update(cur_cm)
        return self._forward(cur_cm)

    def update(self, cur_cm):
        self.conf_mat.update(cm)

    def streaming_value(self):
        final_cm = self.conf_mat.streaming_value()
        return self._forward(final_cm)


class MeanIoU(MetricBase):
    def __init__(self,
                 num_classes,
                 name='mean_iou'):
        super(MeanIoU, self).__init__(name)

        self.iou_per_class = IoUPerClass(num_classes)

    def forward(self, y_pred, y_true):
        v_per_class = self.iou_per_class(y_pred, y_true)

        miou = np.mean(v_per_class)
        return miou

    def update(self):
        pass

    def streaming_value(self):
        v_per_class = self.iou_per_class.streaming_value()
        miou = np.mean(v_per_class)
        return miou


def recall(y_pred, y_true):
    pass


def precision(y_pred, y_true):
    pass


def pr_curve(y_pred, y_true, num_threshold=201):
    pass


if __name__ == '__main__':
    import numpy as np

    cm = ConfusionMatrix(num_classes=2)

    acc = Accuracy()
    pred1 = torch.from_numpy(np.array([1, 1, 0, 1]))
    gt1 = torch.from_numpy(np.array([1, 1, 1, 1]))

    pred2 = torch.from_numpy(np.array([0, 1, 1, 0]))
    gt2 = torch.from_numpy(np.array([1, 1, 1, 1]))
