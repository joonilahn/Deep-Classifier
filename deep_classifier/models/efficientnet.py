from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import (
    get_same_padding_conv2d,
    round_filters,
)

def efficientNet(
    modelname, num_classes, pretrained=None, in_channels=3, use_batchnorm=True
):
    if pretrained:
        model = EfficientNet.from_pretrained(
            modelname,
            num_classes=num_classes,
            in_channels=in_channels,
        )
    else:
        override_params = {"num_classes": num_classes, "use_batchnorm": use_batchnorm}
        model = EfficientNet.from_name(modelname, override_params=override_params)
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size = model._global_params.image_size)
            out_channels = round_filters(32, model._global_params)
            model._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
            
    return model
