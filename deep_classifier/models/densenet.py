import torch
import torch.nn as nn
from torchvision.models.densenet import *


def customDenseNet(num_layers, num_classes, pretrained=False):
    if num_layers == 121:
        model = densenet121(num_classes=num_classes)
    elif num_layers == 161:
        model = densenet161(num_classes=num_classes)
    elif num_layers == 169:
        model = densenet169(num_classes=num_classes)
    elif num_layers == 201:
        model = densenet201(num_classes=num_classes)
    elif num_layers == 264:
        model = DenseNet(
            num_init_features=64,
            growth_rate=32,
            block_config=(6, 12, 64, 48),
            drop_rate=0.2,
            num_classes=num_classes,
        )

    if pretrained:
        state_dict = torch.load(pretrained)
        pretrained_dict = {
            k: v for k, v in state_dict.items() if not k.startswith("classifier")
        }
        model.load_state_dict(pretrained_dict, strict=False)

    return model
