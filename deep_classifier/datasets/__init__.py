from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .dataset import CustomDataset, SubsetRandomSampler, SubsetSampler
from ..transforms import get_train_val_transform


def get_dataloader(cfg, is_train=True):
    batch_size = cfg.SOLVER.BATCH_SIZE_PER_DEVICE * cfg.SYSTEM.NUM_GPUS
    max_data = cfg.DATASETS.MAX_NUM_DATA
    test_size = cfg.DATASETS.TEST_SIZE
    num_workers = cfg.SYSTEM.WORKERS
    pin_memory = cfg.SYSTEM.PIN_MEMORY

    train_transform, val_transform = get_train_val_transform(cfg)

    if is_train:
        train_dataset = CustomDataset(
            cfg.DATASETS.TRAIN,
            train_transform,
            color_map=cfg.DATASETS.COLOR_MAP,
            max_data=max_data,
        )
        val_dataset = CustomDataset(
            cfg.DATASETS.TRAIN,
            val_transform,
            color_map=cfg.DATASETS.COLOR_MAP,
            max_data=max_data,
        )
        num_classes = len(set(train_dataset.labels))

        # train, valid data split (stratified)
        num_data = len(train_dataset)
        labels = train_dataset.labels
        indices = list(range(num_data))
        train_idx, val_idx, train_labels, val_labels = train_test_split(
            indices, labels, test_size=test_size, random_state=42, stratify=labels
        )

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetSampler(val_idx)

        # train, valid dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        dataloaders = {"train": train_loader, "val": val_loader}

        return dataloaders, num_classes

    else:
        test_dataset = CustomDataset(
            cfg.DATASETS.TEST, val_transform, color_map=cfg.DATASETS.COLOR_MAP, is_train=False
        )
        num_classes = len(set(test_dataset.labels))
        num_data = len(test_dataset)
        indices = list(range(num_data))
        test_sampler = SubsetSampler(indices)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            sampler=test_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return test_loader, num_classes
