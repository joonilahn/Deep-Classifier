import torch
import torch.nn as nn
from torchvision.models.resnet import (
    BasicBlock,
    Bottleneck,
    ResNet,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)


def customResNet(
    num_layers, num_classes, pretrained=False, feature_only=True, finetune_from=None
):
    if num_layers == 18:
        model = resnet18(num_classes=num_classes)
    elif num_layers == 34:
        model = resnet34(num_classes=num_classes)
    elif num_layers == 50:
        model = resnet50(num_classes=num_classes)
    elif num_layers == 101:
        model = resnet101(num_classes=num_classes)
    elif num_layers == 152:
        model = resnet152(num_classes=num_classes)

    if pretrained:
        state_dict = torch.load(pretrained)
        if feature_only:
            pretrained_dict = {
                k: v for k, v in state_dict.items() if not k.startswith("fc")
            }
            model.load_state_dict(pretrained_dict, strict=False)
        else:
            model.load_state_dict(state_dict)

    if finetune_from:
        for name, param in model.named_parameters():
            if "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    return model


def resnet18_gray(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model for gray-scale images.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    if pretrained:
        state_dict = torch.load(pretrained)
        model.load_state_dict(state_dict)

    return model
