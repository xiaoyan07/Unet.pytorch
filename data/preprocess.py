import numpy as np

def flip_left_right(image, mask=None):
    ret = []
    new_image = image[:, ::-1, :]
    ret.append(new_image)

    if mask is not None:
        new_mask = mask[:, ::-1, :]
        ret.append(new_mask)
    return tuple(ret)


def flip_up_down(image, mask=None):
    ret = []
    new_image = image[::-1, :, :]
    ret.append(new_image)

    if mask is not None:
        new_mask = mask[::-1, :, :]
        ret.append(new_mask)
    return tuple(ret)


def normalize(image, radius=1):
    n_image = 2 * radius / 255. * image - radius
    return n_image


def denormalize(image, radius=1):
    return (image + radius) * 255. / (2 * radius)
