from typing import Dict, Union
import os
import torch
import numpy as np
import pickle
import random
from multiprocessing import cpu_count

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from imgaug import augmenters as iaa


dataset_names = ["cifar-10", "tiny", "tiny-imagenet-200", "imagenet"]


class CIFAR10Dataset(Dataset):
    def __init__(self, **kwargs):
        self.images = kwargs["images"]
        self.labels = kwargs["labels"]
        print(self.images.__len__())

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        item = torch.from_numpy(self.images[index].reshape((3, 32, 32))).float()
        label = torch.Tensor([self.labels[index]]).type(torch.LongTensor)
        return (item, label)

    @classmethod
    def splits(cls, train_val_split=0.8):
        root = "data/cifar-10-batches-py/"
        images = []
        labels = []
        items = {}
        for i in range(5):
            with open(root + "data_batch_"+str(i+1), 'rb') as fo:
                vals = pickle.load(fo, encoding='bytes')
                images.extend(vals[b'data'])
                labels.extend(vals[b'labels'])
        len = images.__len__()
        indices = list(range(len))
        random.shuffle(indices)
        pivot = int(train_val_split*len)
        items["train"] = {"images": np.array(
            images)[indices[:pivot]], "labels": np.array(labels)[indices[:pivot]]}
        items["val"] = {"images": np.array(
            images)[indices[pivot:]], "labels": np.array(labels)[indices[pivot:]]}
        with open(root + "test_batch", 'rb') as fo:
            vals = pickle.load(fo, encoding='bytes')
            items["test"] = {"images": np.array(
                vals[b'data']), "labels": np.array(vals[b'labels'])}
        train = cls(images=items["train"]["images"], labels=items["train"]["labels"])
        val = cls(images=items["val"]["images"], labels=items["val"]["labels"])
        test = cls(images=items["test"]["images"], labels=items["test"]["labels"])
        return train, val, test


def get_data(dataset_name: str, batch_size: Union[int, Dict[str, int]],
             workers: Union[int, Dict[str, int]], no_transform: bool = False):
    num_cpu = cpu_count()
    if isinstance(batch_size, int):
        print("Only one batch_size given. Rest will be set to 128")
        batch_size = {"train": batch_size, "val": 128, "test": 128}
    if isinstance(workers, int):
        print("Only train workers given. Rest will be set to 0")
        if workers == num_cpu:
            print("Warning: Workers == cpu_count given")
        workers = {"train": max(0, min(workers, num_cpu)), "val": 0, "test": 0}
    else:
        workers = {"train": max(0, min(num_cpu, workers.get("train", 0))),
                   "val": max(0, min(num_cpu, workers.get("val", 0))),
                   "test": max(0, min(num_cpu, workers.get("test", 0)))}
    if dataset_name == "cifar-10":
        mean, std = [0.491, 0.482, 0.447], [0.2023, 0.1994, 0.2010]
        # NOTE: was [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        # which is the same as imagenet which makes me suspicious
        normalize = transforms.Normalize(mean=mean, std=std)
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        if no_transform:
            train_transform = val_transform
        data = {"name": dataset_name,
                "train": datasets.CIFAR10(root='./data', train=True, transform=train_transform,
                                          download=True),
                "val": datasets.CIFAR10(root='./data', train=False, transform=val_transform),
                "test": None}
        dataloaders = {"train": torch.utils.data.DataLoader(data["train"],
                                                            batch_size=batch_size["train"],
                                                            shuffle=True,
                                                            num_workers=workers["train"],
                                                            pin_memory=True),
                       "val": torch.utils.data.DataLoader(data["val"],
                                                          batch_size=batch_size["val"],
                                                          shuffle=False,
                                                          num_workers=workers["val"],
                                                          pin_memory=True),
                       "test": None}
        num_classes = 10
    elif dataset_name == "imagenet":
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)
        train_transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(224, 4, padding_mode="reflect"),
            # transforms.ColorJitter(),
            np.array,
            iaa.Sequential([
                iaa.Fliplr(p=0.5),
                iaa.ContrastNormalization((0.75, 1.5)),
                iaa.Multiply((0.8, 1.2), per_channel=True),
                iaa.Affine(scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)},
                           rotate=(-25, 25),
                           shear=(-8, 8),
                           translate_percent={'x': (-0.2, 0.2), 'y': (-0.2, 0.2)})
            ], random_order=True).augment_image,
            transforms.ColorJitter(),
            transforms.RandomCrop(224, 4, padding_mode="reflect"),
            transforms.ToTensor(),
            normalize
        ])
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize])
        if no_transform:
            train_transform = val_transform
        data = {"name": dataset_name,
                'train': datasets.ImageNet(root='./data/imagenet/ILSVRC',
                                           split="train",
                                           transform=train_transform,
                                           download=False),
                'val': datasets.ImageNet(root='./data/imagenet/ILSVRC',
                                         split="val",
                                         transform=val_transform),
                'test': None}
        dataloaders = {"train": torch.utils.data.DataLoader(data["train"],
                                                            batch_size=batch_size["train"],
                                                            shuffle=True,
                                                            num_workers=workers["train"],
                                                            pin_memory=True),
                       "val": torch.utils.data.DataLoader(data["val"],
                                                          batch_size=batch_size["val"],
                                                          shuffle=False,
                                                          num_workers=workers["val"],
                                                          pin_memory=True),
                       "test": None}
        num_classes = 1000
    elif "tiny" in dataset_name:
        # NOTE: This I think we had calculated ourselves
        mean, std = [0.481, 0.448, 0.397], [0.276, 0.268, 0.282]
        normalize = transforms.Normalize(mean=mean, std=std)
        train_transform = transforms.Compose([
            np.array,
            iaa.Sequential([
                iaa.Fliplr(p=0.5),
                iaa.ContrastNormalization((0.75, 1.5)),
                iaa.Multiply((0.8, 1.2), per_channel=True),
                iaa.Affine(scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)},
                           rotate=(-25, 25),
                           shear=(-8, 8),
                           translate_percent={'x': (-0.2, 0.2), 'y': (-0.2, 0.2)})
            ], random_order=True).augment_image,
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomResizedCrop(64, 4, padding_mode="reflect"),
            transforms.ToTensor(),
            normalize
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        if no_transform:
            train_transform = val_transform
        data_transforms = {
            'train': train_transform,
            'val': val_transform,
            'test': val_transform
        }
        data_dir = './data/tiny-imagenet-200/'
        data = {x: datasets.ImageFolder(os.path.join(
            data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test']}
        data["name"] = "tiny-imagenet-200"
        dataloaders = {x: torch.utils.data.DataLoader(
            data[x], batch_size=batch_size[x], shuffle=True,
            num_workers=workers[x]) for x in ['train', 'val']}
        dataloaders["test"] = None
        num_classes = 200
    else:
        raise ValueError("Unknown value for data")
    return data, dataloaders, num_classes


def get_data_for_label(dataset_name: str, dataset, label: int, batch_size: int):
    if dataset_name in {"cifar-10", "tiny", "tiny-imagenet-200"}:
        instances = [x[0] for x in dataset if x[1] == label]
        if batch_size:
            batches = []
            j = 0
            batch = instances[j * batch_size: (j+1) * batch_size]
            while batch:
                batches.append(torch.stack(batch))
                j += 1
                batch = instances[j * batch_size: (j+1) * batch_size]
        else:
            batches = instances
    elif dataset_name == "imagenet":
        indices = [i for i, x in enumerate(dataset.samples) if x[1] == label]
        instances = []
        for i in indices:
            instances.append(dataset[i][0])
        if batch_size:
            batches = []
            j = 0
            batch = instances[j * batch_size: (j+1) * batch_size]
            while batch:
                batches.append(torch.stack(batch))
                j += 1
                batch = instances[j * batch_size: (j+1) * batch_size]
        else:
            batches = instances
    return batches
