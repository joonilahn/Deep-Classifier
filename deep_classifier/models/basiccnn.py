import torch
import torch.nn as nn

basic_cfg = [32, "M", 64, "M", 128, 128, "M", 256, 256, "M", 256, 256, "M"]
super_basic_cfg = [8, "M", 16, "M", 32, 32, "M"]
# super_basic_cfg2 = [16, "M", 32, "M", 64, 64, "M", 128, 128, "M"]
# super_basic_cfg3 = [8, "M", 16, 16, "M"]

class BasicCNN(nn.Module):
    def __init__(self, features, num_classes=2, init_weights=True):
        super(BasicCNN, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class SuperBasicCNN(nn.Module):
    def __init__(self, features, num_classes=2, init_weights=True):
        super(SuperBasicCNN, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        num_hidden = 1024
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, num_hidden),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(num_hidden, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

                
def make_layers(cfg, in_channels=1, batch_norm=False):
    layers = []
    in_channels = in_channels
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def basic_cnn(num_classes=2, cfg=basic_cfg, pretrained=None):
    model = BasicCNN(make_layers(cfg), num_classes=num_classes)
    if pretrained:
        model.load_state_dict(torch.load(pretrained))
        print("Loaded %s" % pretrained)

    return model

def super_basic_cnn(num_classes=2, cfg=super_basic_cfg, pretrained=None):
    model = SuperBasicCNN(make_layers(cfg, in_channels=3), num_classes=num_classes)
    if pretrained:
        model.load_state_dict(torch.load(pretrained))
        print("Loaded %s" % pretrained)

    return model

