import torch
from data import preprocess
import cv2


def test_func():
    pass


class PredictPack(object):
    def __init__(self,
                 model,
                 use_gpu=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.model = model

    def im_predict(self, inputs):
        inputs = inputs.to(self.device)

        output = self.model(inputs)

        return output

    def im_predict_flip_up_down(self, inputs):
        fliped_inputs = preprocess.flip_up_down(inputs)

        fliped_outputs = self.im_predict(fliped_inputs)
        return fliped_outputs

    def im_predict_flip_left_right(self, inputs):
        fliped_inputs = preprocess.flip_left_right(inputs)
        fliped_outputs = self.im_predict(fliped_inputs)
        return fliped_outputs

    def im_predict_rotate(self, inputs, angle=90):
        """

        Args:
            inputs:
            angle: rotating radius

        Returns:

        """
        pass

    def im_predict_distort_color(self, inputs):
        pass

    def im_predict_scale(self, inputs, scale_factor=1.25):
        resized_inputs = cv2.resize(inputs, None, None, fx=scale_factor, fy=scale_factor,
                                    interpolation=cv2.INTER_LINEAR)

        resized_outputs = self.im_predict(resized_inputs)

        return resized_outputs
