import torch
import torch.utils.data as data
import glob
import os
import logging
from skimage.io import imread

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SegData(data.Dataset):
    def __init__(self,
                 image_dir,
                 mask_dir,
                 image_format='jpg',
                 mask_format='png',
                 training=True,
                 image_mask_mapping=None):
        super(SegData, self).__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.im_format = image_format
        self.mask_format = mask_format
        self.training = training
        self.image_mask_mapping = image_mask_mapping
        self.idx_to_im_mask_path = self._build_index()

    def _build_index(self):
        im_path_list = glob.glob(os.path.join(self.image_dir, '*.{}'.format(self.im_format)))
        if len(im_path_list) == 0:
            raise FileNotFoundError('The image is not found.')
        if self.image_mask_mapping is None:
            mask_path_list = [
                os.path.join(self.mask_dir, os.path.split(im_path)[-1].replace(self.im_format, self.mask_format)) for
                im_path in
                im_path_list]
        else:
            mask_path_list = [os.path.join(self.mask_dir, self.image_mask_mapping(os.path.split(im_path)[-1])) for
                              im_path in
                              im_path_list]
        idx_to_im_mask_path = list(map(lambda e: (e[0], e[1]), zip(im_path_list, mask_path_list)))
        return idx_to_im_mask_path

    def _preprocess(self, image, mask):

        return image, mask

    def __getitem__(self, idx):
        im_path, mask_path = self.idx_to_im_mask_path[idx]
        im = imread(im_path)
        mask = imread(mask_path)

        im, mask = self._preprocess(im, mask)

        im_ts = torch.from_numpy(im)
        mask_ts = torch.from_numpy(mask)
        return im_ts, mask_ts

    def __len__(self):
        raise NotImplementedError


if __name__ == '__main__':
    a = [1, 9, 8, 7, 6]
    b = [1, 2, 3, 4, 5]
    c = list(map(lambda e: (e[0], e[1]), zip(a, b)))
    print(c)
