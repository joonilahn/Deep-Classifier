import random

import torch
from PIL import Image
from torchvision.transforms import functional as F

all = [
    "UpperLeftCrop",
    "UpperRightCrop",
    "BottomLeftCrop",
    "InvertColor",
    "RandomResize",
    "ExpandTensorCH",
    "SwapImageRatio",
    "RandomSwapImageRatio",
    "RandomRotation",
    "RightBottomCrop"
]

class RightBottomCrop(object):
    """
    Crop right-bottom part of the image
    """
    def __init__(self):
        pass
    
    def get_croparea(self, img):
        w, h = img.size
        left = w // 2
        right = w
        up = h - w // 2
        down = h
        return (left, up, right, down)
    
    def make_crop(self, img):
        crop_area = self.get_croparea(img)
        img_cropped = img.crop(crop_area)
        
        return img_cropped
    
    def __call__(self, img):
        return self.make_crop(img)
    
    
class UpperLeftCrop(object):
    """
    Crop Upper-Left part of the image
    """

    def __init__(self, ratio=0.33):
        self.ratio = ratio

    def get_croparea(self, img):
        w, h = img.size
        shorter_size = min(w, h)
        left = 0
        right = int(shorter_size * self.ratio)
        up = 0
        down = int(shorter_size * self.ratio)

        return (left, up, right, down)

    def make_crop(self, img):
        crop_area = self.get_croparea(img)
        img_leftupper = img.crop(crop_area)

        return img_leftupper

    def __call__(self, img):
        return self.make_crop(img)


class UpperRightCrop(object):
    """
    Crop Upper-Right part of the image
    """

    def __init__(self, ratio=0.33):
        self.ratio = ratio

    def get_croparea(self, img):
        w, h = img.size
        shorter_size = min(w, h)
        left = w - int(shorter_size * self.ratio)
        right = w
        up = 0
        down = int(shorter_size * self.ratio)

        return (left, up, right, down)

    def make_crop(self, img):
        crop_area = self.get_croparea(img)
        img_upperright = img.crop(crop_area)

        return img_upperright

    def __call__(self, img):
        return self.make_crop(img)


class BottomLeftCrop(object):
    """
    Crop Bottom-Left part of the image
    """

    def __init__(self, ratio=0.33):
        self.ratio = ratio

    def get_croparea(self, img):
        w, h = img.size
        shorter_size = min(w, h)
        left = 0
        right = int(shorter_size * self.ratio)
        up = h - int(shorter_size * self.ratio)
        down = h

        return (left, up, right, down)

    def make_crop(self, img):
        crop_area = self.get_croparea(img)
        img_bottomleft = img.crop(crop_area)

        return img_bottomleft

    def __call__(self, img):
        return self.make_crop(img)


class InvertColor(object):
    """
    Invert color of images
    Black -> White / White -> Black
    """

    def __init__(self):
        pass

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be inverted

        Returns:
            PIL Image: Inverted image
        """
        img = -img + 1
        return img

    def __repr__(self):
        return self.__class__.__name__ + " invert all channels"


class RandomResize(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size_range, interpolation=Image.BILINEAR):
        assert isinstance(size_range, list) or isinstance(size_range, tuple)
        assert len(size_range) == 2
        self.size_range = size_range

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        size = random.randint(224, 384)
        return F.resize(img, size)


class ExpandTensorCH:
    """Expand a image channel of a tensor.

    Args:
        image(Torch.Tensor): image to be expanded
        
    Returns:
        image(Torch.Tensor): Expanded image
    """

    def __init__(self):
        pass

    def __call__(self, img, in_channels=1, out_channels=3):
        n_dims = len(img.size())
        assert n_dims == 2 or n_dims == 3

        if n_dims == 2:
            img = img.unsqueeze(0)
        elif n_dims == 3:
            assert img.size()[0] == in_channels

        return torch.cat([img] * out_channels, dim=0)


class SwapImageRatio:
    """
    Swap width and height of an input image
    img size: (w, h) -> (h, w)
    
    Args:
        image(PIL Image): image to be expanded
        
    Returns:
        image(PIL Image): Expanded image
    """

    def __init__(self):
        pass

    def __call__(self, img):
        return img(img.size[::-1])


class RandomSwapImageRatio:
    """
    Swap width and height of an input image
    img size: (w, h) -> (h, w)
    
    Args:
        image(PIL Image): image to be expanded
        
    Returns:
        image(PIL Image): Expanded image
    """

    def __init__(self):
        pass

    def __call__(self, img):
        return img(img.size[::-1])

    def _swap_img_ratio(self, img):
        return img(img.size[::-1])


class RandomRotation:
    """
    Rotate the given PIL Image randomly with a given angle and a probability.
    
    Args:
        int (float): angle to rotate the image. 
        p (float): probability of the image being flipped. Default value is 0.5
        expand (bool): True if the transformed image should be expanded 
                        as the new size of the image
    """

    def __init__(self, p):
        self.angle = (90, 180, 270)
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be Perspectively transformed.
        Returns:
            PIL Image: Random rotated image.
        """
        if not F._is_pil_image(img):
            raise TypeError("img should be PIL Image. Got {}".format(type(img)))

        if random.random() < self.p:
            rotation_angle = random.sample(self.angle, 1)[0]
            return img.rotate(rotation_angle, expand=True)
        return img
