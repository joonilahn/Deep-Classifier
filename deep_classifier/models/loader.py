from .basiccnn import basic_cnn, super_basic_cnn
from .densenet import customDenseNet
from .inception import customInception
from .inceptionv4 import inception_v4
from .mobilenet import mobilenet
from .pnasnet import pnasnet5
from .resnet import customResNet, resnet18_gray
from .resnext import resnext
from .efficientnet import efficientNet
import re

def load_model(
    modelname, num_classes, pretrained, batch_norm=True, finetune_from=False
):
    if modelname.lower() == "inception-v4":
        print("Loading %s" % modelname)
        model = inception_v4(num_classes=num_classes, pretrained=pretrained)
        return model

    elif modelname.lower() == "inception-v3":
        print("Loading %s" % modelname)
        model = customInception(num_classes=num_classes, pretrained=pretrained)
        return model

    elif modelname.lower() == "pnasnet5":
        print("Loading %s" % modelname)
        model = pnasnet5(
            num_classes=num_classes, pretrained=pretrained, finetune_from=finetune_from
        )
        return model

    elif "resnet" in modelname.lower():
        print("Loading %s" % modelname)

        if "gray" in modelname.lower():
            model = resnet18_gray(num_classes=num_classes, pretrained=pretrained)
        else:
            num_layers = int(re.search("\d+", modelname).group())
            model = customResNet(
                num_layers=num_layers,
                num_classes=num_classes,
                pretrained=pretrained,
                finetune_from=finetune_from,
            )
        return model

    elif "resnext" in modelname.lower():
        print("Loading %s" % modelname)
        model = resnext(
            num_classes=num_classes,
            modelname=modelname,
            pretrained=pretrained,
            finetune_from=finetune_from,
        )
        return model

    elif "densenet" in modelname.lower():
        print("Loading %s" % modelname)
        num_layers = int(re.search("\d+", modelname).group())
        model = customDenseNet(num_layers, num_classes, pretrained=pretrained)
        return model

    elif modelname.lower() == "mobilenet":
        print("Loading %s" % modelname)
        model = mobilenet(num_classes=num_classes, pretrained=pretrained)
        return model

    elif modelname.lower() == "basiccnn":
        print("Loading %s" % modelname)
        model = basic_cnn(num_classes=num_classes, pretrained=pretrained)
        return model
    
    elif modelname.lower() == "superbasiccnn":
        print("Loading %s" % modelname)
        model = super_basic_cnn(num_classes=num_classes, pretrained=pretrained)
        return model

    elif "efficientnet" in modelname.lower():
        """
        modelname: efficientnet-b{number}
        e.g) efficientnet-b6
        """
        print("Loading %s" % modelname)
        if modelname.endswith("gray"):
            modelname = re.search(r"(efficientnet-b\d{1}).*", modelname).group(1)
            model = efficientNet(
                modelname,
                num_classes,
                in_channels=1,
                pretrained=pretrained,
                use_batchnorm=batch_norm,
            )
        else:
            model = efficientNet(
                modelname, num_classes, pretrained=pretrained, use_batchnorm=batch_norm
            )
        return model

    else:
        raise NotImplementedError
