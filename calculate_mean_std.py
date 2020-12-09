import os
import sys

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from datasets.dataset import CustomDataset
from transforms.custom_transforms import InvertColor

def online_mean_and_sd(loader):
    """Compute the mean and sd in an online fashion

        Var[x] = E[X^2] - E^2[X]
    """
    cnt = 0
    use_gpu = True
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    if use_gpu:
        fst_moment = fst_moment.cuda()
        snd_moment = snd_moment.cuda()

    for i, data in enumerate(loader):
        imgs = data[0]
        imgs *= 255.0
        if use_gpu:
            imgs = imgs.cuda()
        b, c, h, w = imgs.shape
        nb_pixels = b * h * w
        sum_ = imgs.sum(dim=0).sum(dim=-1).sum(dim=-1)
        sum_of_square = (imgs ** 2).sum(dim=0).sum(dim=-1).sum(dim=-1)
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels
        print("Calculated batch {}/{}".format(i + 1, len(loader)), end="\r")

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)


if __name__ == "__main__":
    maindir = sys.argv[1]
    outputfile = sys.argv[2]
    train_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor()]
    )
    dataset = CustomDataset(os.path.abspath(sys.argv[1]), transform=train_transform)
    loader = DataLoader(dataset, batch_size=1600, pin_memory=False)

    mean, std = online_mean_and_sd(loader)

    print("Mean: {}, Std: {}".format(mean.data, std.data))
    with open(outputfile, "w") as f:
        f.write("Mean: {}, Std: {}".format(mean.data, std.data))
