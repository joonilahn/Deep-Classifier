import argparse
import os
import warnings

import torch
import torch.nn as nn

from deep_classifier.config.defaults import get_cfg_defaults
from deep_classifier.datasets import get_dataloader
from deep_classifier.logger.logger import CreateLogger
from deep_classifier.models import load_model

warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", UserWarning)


# Parse console arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config-file", type=str, help="yaml type config file to be used for training"
)
parser.add_argument("--output-file", type=str, help="a path for the output log file",
                   default='test_result.txt')
parser.add_argument("--weight", type=str, help="trained weight to be used in the test")
parser.add_argument(
        "--opts",
        help="Modify config options using the command-line",
        default=[],
        nargs=argparse.REMAINDER,
    )
args = parser.parse_args()

# Get configs from a config file
CFG = get_cfg_defaults()
CFG.merge_from_file(args.config_file)
CFG.merge_from_list(args.opts)

device_ids = ",".join(str(d) for d in CFG.SYSTEM.DEVICE_IDS)
os.environ["CUDA_VISIBLE_DEVICES"] = device_ids

# Create system logger
MY_LOGGER = CreateLogger("deep-classifier", args.output_file)
MY_LOGGER.info(CFG)

def test_model(model, testloader):
    model.eval()
    running_corrects = 0
    num_data = 0
    MY_LOGGER.info("Starting Test")
    with torch.no_grad():
        for i, data in enumerate(testloader):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            # forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)           
            running_corrects += torch.sum(preds == labels.data).item()
            num_data += inputs.size(0)
            MY_LOGGER.info(
                "Test iterations: {0:}/{1:}".format((i + 1), len(testloader))
            )

    test_acc = running_corrects / num_data * 100.0
    MY_LOGGER.info(
        "\nTest Accuracy for the {:d} test images: {:.2f}%".format(
            num_data, test_acc
        )
    )

    return test_acc


def main():
    # load dataloader
    test_loader, num_classes = get_dataloader(CFG, is_train=False)
    MY_LOGGER.info("Number of classes for the dataset is %d" % num_classes)

    # load model
    model = load_model(CFG.MODEL.BACKBONE, num_classes, None)

    # load weight
    weight = torch.load(args.weight)
    
    try:
        model.load_state_dict(weight)
    except:
        class WrappedModel(nn.Module):
            def __init__(self, module):
                super(WrappedModel, self).__init__()
                self.module = module
            def forward(self, x):
                return self.module(x)
        model = WrappedModel(model)
        model.load_state_dict(weight)
    print("Loaded %s" % args.weight)

    # move the model to cuda tensor if use_gpu is true
    model = model.cuda()

    if CFG.SYSTEM.MULTI_GPU:
        MY_LOGGER.info("Using DataParallel")
        model = torch.nn.DataParallel(model)

    # Train the model
    test_model(model, test_loader)


if __name__ == "__main__":
    main()
