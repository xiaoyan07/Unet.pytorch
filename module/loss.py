import torch
import torch.nn.functional as F
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self,
                 smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def _dice_coeff(self, pred, target):
        """

        Args:
            pred: [N, 1]
            target: [N, 1]

        Returns:

        """

        smooth = self.smooth
        inter = torch.sum(pred * target)
        z = pred.sum() + target.sum() + smooth
        return (2 * inter + smooth) / z

    def forward(self, pred, target):
        return 1. - self._dice_coeff(pred, target)


class BinarySegmentationLoss(nn.Module):
    """ Dice + BCE

    """

    def __init__(self):
        super(BinarySegmentationLoss, self).__init__()

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()

    def forward(self, pred, target):
        bce_loss_v = self.bce_loss(pred, target)
        dice_loss_v = self.dice_loss(pred, target)

        total = bce_loss_v + dice_loss_v
        return total


if __name__ == '__main__':
    a = torch.ones(3, 5)
    b = a.sum()
    c = torch.sum(a)

    print(b)
    print(c)
