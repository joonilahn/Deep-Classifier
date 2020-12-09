import glob
import os
import random

import numpy as np

# from sklearn.model_selection import train_test_split
from PIL import Image, ImageFile
from torch.utils.data import Dataset

# from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from torchvision import transforms
from torchvision.transforms import functional as F

ImageFile.LOAD_TRUNCATED_IMAGES = True
random.seed(42)


class CustomDataset(Dataset):
    def __init__(
        self,
        root_dir,
        transform,
        color_map="RGB",
        subdir=None,
        get_errors=False,
        max_data=None,
        is_train=True
    ):
        """
        Args:
            root_dir (string): Directory containing mutiple subdirectories. Each subdirectory contains images for each label.
        """
        super(CustomDataset, self).__init__()
        if subdir:
            self.root_dir = os.path.join(root_dir, subdir)
        else:
            self.root_dir = root_dir
        self.get_errors = get_errors
        
        if not is_train:
            self.max_data = None
        elif max_data > 0:
            self.max_data = max_data
        else:
            self.max_data = None
            
        self.color_map = color_map
        self.transform = transform
        self.labellist = [
            l for l in os.listdir(self.root_dir) if l != ".ipynb_checkpoints"
        ]
        self._labeldict = {}
        self.labels = []
        self.files = []
        self.support_ext = ("jpg", "jpeg", "png", "tif", "tiff", "gif", "bmp")
        self.sort_labels()
        self.get_labels()

    def sort_labels(self):
        self.labellist = [
            l for l in self.labellist if os.path.isdir(os.path.join(self.root_dir, l))
        ]
        self.labellist.sort()

    def get_labels(self):
        for i, label in enumerate(self.labellist):
            labelpath = os.path.join(self.root_dir, label)
            files_i = os.listdir(labelpath)
            if self.max_data:
                random.shuffle(files_i)
                files_i = files_i[: self.max_data]
            files_i = [
                os.path.join(labelpath, f)
                for f in files_i
                if f.split(".")[1].lower() in self.support_ext
            ]
            self.files += files_i
            self.labels += [i] * len(files_i)
            self._labeldict[i] = label

    @property
    def labeldict(self):
        return self._labeldict

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        try:
            image = Image.open(img_name)
            label = self.labels[idx]
        except:
            return

        image = image.convert(mode=self.color_map)

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        if self.get_errors:
            return (image, label, img_name)
        else:
            return (image, label)


class SubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

if __name__ == "__main__":
    import sys

    root_dir = sys.argv[1]
    dataloaders, num_classes = get_dataloader(root_dir, batch_size=2)
    print("num classes is %d" % num_classes)
