import os
import pathlib

import torch
from tensorboardX import SummaryWriter


def savelogs(logger, phase, epoch, loss, accuracy, global_iterations):
    info = {"epoch": epoch, phase + " loss": loss, phase + " accuracy": accuracy}
    for tag, value in info.items():
        logger.add_scalar(tag, value, global_iterations)


def make_dir(dirpath):
    pathlib.Path(dirpath).mkdir(parents=True, exist_ok=True)


def save_model(model, optimizer, epoch, dirpath, filename):
    make_dir(dirpath)
    filepath = os.path.join(dirpath, filename)
    state_dict = model.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()
    
    torch.save(
        {
            "epoch": epoch, 
            "state_dict": state_dict, 
            "optimizer": optimizer,
        }, 
        filepath
    )


def save_params(best_model_wts, dirpath, filename):
    make_dir(dirpath)
    filepath = os.path.join(dirpath, filename)
    torch.save(best_model_wts, filepath)


def save_configs(dirpath, filename, configfile, configs_to_save):
    make_dir(dirpath)
    with open(configfile, "r") as f:
        configs_str = f.read()
    write_path = os.path.join(dirpath, filename)
    with open(write_path, "w") as f:
        for k, v in configs_to_save.items():
            f.write(k + ": " + str(v) + "\n")
