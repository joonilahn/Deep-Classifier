import argparse

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from config.defaults import get_cfg_defaults
from models import load_model
from transforms import get_test_transform

# Parse console arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config-file", type=str, help="yaml type config file to be used for test"
)
parser.add_argument("--test-folder", type=str, help="file to be used in inference")
parser.add_argument("--weight", type=str, help="weight file to be used in the test")
parser.add_argument("--file", type=str, help="file to be inferenced")
args = parser.parse_args()


class Classifier:
    def __init__(self, weight, cfg, label_dict, num_classes=2, device="cuda:0"):
        self.model = load_model(cfg.MODEL.BACKBONE, num_classes, None)
        self.num_classes = num_classes
        self.cfg = cfg
        self.device = device
        self.label_dict = label_dict

        # transform
        self.transform = get_test_transform(cfg)

        # load pretrained model
        self.model = self._load_model(self.model, weight)

        # load the model into device
        self.model = self.model.to(device)

    def _load_model(self, model, weight):
        # load pretrained weight
        print("loading pretrained model from %s" % weight)
        weights = torch.load(weight)
        try:
            model.load_state_dict(weights)
        except:
            model = WrappedModel(model)
            model.load_state_dict(weights)
        print("loaded recognizer weight")

        return model

    def _read_image(self, file):
        try:
            img = Image.open(file)
        except:
            raise OSError(file)

        return img.convert(mode=self.cfg.DATASETS.COLOR_MAP)

    def _transform_image(self, img):
        return self.transform(img)

    @torch.no_grad()
    def inference(self, file):
        # read image
        img = self._read_image(file)

        # transform image
        img = self._transform_image(img)

        # expand dimension of torch image from 3 to 4
        if len(img.size()) == 3:
            img = img.unsqueeze(0)

        # load into device
        img = img.to(self.device)

        # forward pass
        out = self.model(img)

        # prediction
        _, pred = torch.max(out.data, 1)
        pred = int(pred.detach().cpu().data)

        return self.label_dict[pred]


class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module

    def forward(self, x):
        return self.module(x)


if __name__ == "__main__":
    # Get configs from a config file
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)

    labeldict = {0: "licenseplate", 1: "mileage"}
    solver = Classifier(args.weight, cfg, labeldict)
    pred = solver.inference(args.file)
    print(pred)
