from torchvision.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)

from .custom_transforms import *


def get_test_transform(cfg):
    transforms_dict = {
        "Resize": Resize((cfg.DATASETS.IMG_HEIGHT, cfg.DATASETS.IMG_WIDTH)),
        "RandomHorizontalFlip": RandomHorizontalFlip(0.5),
        "ToTensor": ToTensor(),
        "Normalize": Normalize(mean=cfg.DATASETS.MEAN, std=cfg.DATASETS.STD),
        "InvertColor": InvertColor(),
        "UpperLeftCrop": UpperLeftCrop(),
        "UpperRightCrop": UpperRightCrop(),
        "BottomLeftCrop": BottomLeftCrop(),
        "ExpandTensorCH": ExpandTensorCH(),
        "RightBottomCrop": RightBottomCrop()
    }
    test_transform_list = []

    # get train_transform_list
    for transform_type in cfg.DATASETS.TEST_TRANSFORM_TYPES:
        test_transform_list.append(transforms_dict[transform_type])

    return Compose(test_transform_list)


def get_train_val_transform(cfg):
    """
    Define how images are transformed before feeding into a model.
    
    Args:
      - transforms_types(list(str))
    """
    transforms_dict = {
        "Resize": Resize((cfg.DATASETS.IMG_HEIGHT, cfg.DATASETS.IMG_WIDTH)),
        "RandomHorizontalFlip": RandomHorizontalFlip(0.5),
        "ToTensor": ToTensor(),
        "Normalize": Normalize(mean=cfg.DATASETS.MEAN, std=cfg.DATASETS.STD),
        "InvertColor": InvertColor(),
        "UpperLeftCrop": UpperLeftCrop(),
        "UpperRightCrop": UpperRightCrop(),
        "BottomLeftCrop": BottomLeftCrop(),
        "ExpandTensorCH": ExpandTensorCH(),
        "RandomSwapImageRatio": RandomSwapImageRatio(),
        "RandomRotation": RandomRotation(cfg.DATASETS.RANDOM_ROTATION),
        "RightBottomCrop": RightBottomCrop()
    }

    train_transform_list = []
    val_transform_list = []

    # get train_transform_list
    for transform_type in cfg.DATASETS.TRAIN_TRANSFORM_TYPES:
        train_transform_list.append(transforms_dict[transform_type])

    # get val_transform_list
    for transform_type in cfg.DATASETS.TEST_TRANSFORM_TYPES:
        val_transform_list.append(transforms_dict[transform_type])

    # define transform
    train_transform = Compose(train_transform_list)
    val_transform = Compose(val_transform_list)
    train_val_transforms = (train_transform, val_transform)

    return train_val_transforms
