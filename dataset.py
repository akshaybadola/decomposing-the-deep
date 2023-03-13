from typing import Dict, Union, Optional, Callable
import os
import pickle
import random
from multiprocessing import cpu_count

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from imgaug import augmenters as iaa


dataset_names = ["cifar-10", "tiny", "tiny-imagenet-200", "imagenet"]


def test_func(x):
    import ipdb; ipdb.set_trace()
    return x


class ImageDataset(Dataset):
    def __init__(self, images, targets, in_memory, transform=None):
        self.images = images
        self.targets = targets
        self.in_memory = in_memory
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        if self.in_memory:
            img = self.images[i]
        else:
            img = Image.open(self.images[i])
            if img.mode != "RGB":
                img = img.convert("RGB")
        if self.transform:
            return self.transform(img), self.targets[i]
        else:
            return img, self.targets[i]



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
             workers: Union[int, Dict[str, int]], no_transform: bool = False,
             train_sampler: Optional[Callable] = None):
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
        # NOTE: Original was [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        # which is the same as imagenet which makes me suspicious
        # NOTE: Values which I used:
        mean, std = [0.491, 0.482, 0.447], [0.247, 0.243, 0.261]
        # NOTE: Values given by Yash
        # mean, std = [0.491, 0.482, 0.447], [0.2023, 0.1994, 0.2010]
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
        data_dir = "./data"
        data_keys = {"train": True, "val": True, "test": False}
        train_keys = ["train"]
        data_transforms = {'train': train_transform,
                           'val': val_transform,
                           'test': val_transform}
        data = {k: datasets.CIFAR10(root=data_dir, train=True if k == "train" else False,
                                    transform=data_transforms[k], download=True)
                if v else None
                for k, v in data_keys.items()}
        # data = {"name": dataset_name,
        #         "train": datasets.CIFAR10(root='./data', train=True, transform=train_transform,
        #                                   download=True),
        #         "val": datasets.CIFAR10(root='./data', train=False, transform=val_transform),
        #         "test": None}
        # if train_sampler:
        #     ts = train_sampler(range(len(data["train"])), False)
        #     trainloader = torch.utils.data.DataLoader(data["train"],
        #                                               batch_size=batch_size["train"],
        #                                               sampler=ts,
        #                                               num_workers=workers["train"],
        #                                               pin_memory=True)
        # else:
        #     trainloader = torch.utils.data.DataLoader(data["train"],
        #                                               batch_size=batch_size["train"],
        #                                               shuffle=True,
        #                                               num_workers=workers["train"],
        #                                               pin_memory=True)
        # dataloaders = {"train": trainloader,
        #                "val": torch.utils.data.DataLoader(data["val"],
        #                                                   batch_size=batch_size["val"],
        #                                                   shuffle=False,
        #                                                   num_workers=workers["val"],
        #                                                   pin_memory=True),
        #                "test": None}
        num_classes = 10
    elif dataset_name == "imagenet":
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)
        train_transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(224, 4, padding_mode="reflect"),
            # transforms.ColorJitter(),
            np.array,
            # test_func,
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
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            # test_func,
            transforms.RandomCrop(224, 4, padding_mode="reflect"),
            normalize
        ])
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize])
        if no_transform:
            train_transform = val_transform
        data_keys = {"train": True, "val": True, "test": False}
        train_keys = ["train"]
        data_transforms = {'train': train_transform,
                           'val': val_transform,
                           'test': val_transform}
        data = {x: datasets.ImageNet(root='./data/imagenet/ILSVRC',
                                     split=x,
                                     transform=data_transforms[x],
                                     download=False)
                if y else None
                for x, y in data_keys.items()}
        # if train_sampler:
        #     ts = train_sampler(range(len(data["train"])), False)
        #     trainloader = torch.utils.data.DataLoader(data["train"],
        #                                               batch_size=batch_size["train"],
        #                                               sampler=ts,
        #                                               num_workers=workers["train"],
        #                                               pin_memory=True)
        # else:
        #     trainloader = torch.utils.data.DataLoader(data["train"],
        #                                               batch_size=batch_size["train"],
        #                                               shuffle=True,
        #                                               num_workers=workers["train"],
        #                                               pin_memory=True)
        # dataloaders = {"train": trainloader,
        #                "val": torch.utils.data.DataLoader(data["val"],
        #                                                   batch_size=batch_size["val"],
        #                                                   shuffle=False,
        #                                                   num_workers=workers["val"],
        #                                                   pin_memory=True),
        #                "test": None}
        num_classes = 1000
    elif dataset_name == "imagenet12":
        mean, std = [0.481, 0.458, 0.408], [0.268, 0.261, 0.275]
        normalize = transforms.Normalize(mean=mean, std=std)
        train_transform = transforms.Compose([
            np.array,
            # test_func,
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
            transforms.ToTensor(),
            # test_func,
            transforms.RandomCrop(72, 4, padding_mode="reflect"),
            transforms.Resize((72, 72)),
            normalize
        ])
        val_transform = transforms.Compose([
            transforms.Resize((72, 72)),
            transforms.ToTensor(),
            normalize])
        if no_transform:
            train_transform = val_transform
        data_keys = {"train": True, "val": True, "test": False}
        train_keys = ["train"]
        data_transforms = {'train': train_transform,
                           'val': val_transform,
                           'test': val_transform}
        data = {x: datasets.ImageFolder(root=f'./data/imagenet12/{x}',
                                        transform=data_transforms[x])
                if y else None
                for x, y in data_keys.items()}
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
        data_transforms = {'train': train_transform,
                           'val': val_transform,
                           'test': val_transform}
        data_dir = './data/tiny-imagenet-200/'
        data_keys = {"train": True, "val": True, "test": False}
        train_keys = ["train"]
        data = {k: datasets.ImageFolder(os.path.join(data_dir, k), data_transforms[k])
                if v else None
                for k, v in data_keys.items()}
        num_classes = 200
    elif dataset_name == "cub":
        data_dir = "./data/cub"
        img_size = 224
        # CHECK: verify shape and mean, though img_size is the one in the script
        #        for ProtoTree
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)
        train_transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.RandomOrder([
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                transforms.ColorJitter((0.6, 1.4), (0.6, 1.4), (0.6, 1.4), (-0.02, 0.02)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=10, shear=(-2, 2), translate=[0.05, 0.05])
            ]),
            transforms.ToTensor(),
            normalize,
        ])
        test_transform = transforms.Compose([
            transforms.Resize(size=img_size),
            transforms.ToTensor(),
            normalize
        ])
        if no_transform:
            train_transform = test_transform
        data_keys = {'train_corners': True, 'train_crop': True, 'test_full': True}
        train_keys = ["train_corners"]
        val_keys = ["train_crop", "test_full"]
        data_transforms = {k: train_transform if k == "train_corners" else test_transform
                           for k, v in data_keys.items()}
        data = {k: datasets.ImageFolder(os.path.join(data_dir, k), data_transforms[k])
                if v else None
                for k, v in data_keys.items()}
        classes = data["train_corners"].classes
        for i in range(len(classes)):
            classes[i] = classes[i].split('.')[1]
        num_classes = len(classes)
        data["classes"] = classes
    else:
        raise ValueError(f"Unknown value for data {dataset_name}")
    _batch_size = {}
    _workers = {}
    for k, v in data.items():
        if v is not None:
            if k in train_keys:
                _batch_size[k] = batch_size["train"]
                _workers[k] = workers["train"]
            elif k in batch_size and k in workers:
                _batch_size[k] = batch_size[k]
                _workers[k] = workers[k]
            else:
                try:
                    if k in val_keys:
                        _batch_size[k] = batch_size["val"]
                        _workers[k] = workers["val"]
                except Exception:
                    if k in test_keys:
                        _batch_size[k] = batch_size["test"]
                        _workers[k] = workers["test"]
    data["name"] = dataset_name
    batch_size = _batch_size
    workers = _workers
    if train_sampler:
        dataloaders = {x: torch.utils.data.DataLoader(
            data[x], batch_size=batch_size[x],
            sampler=train_sampler(range(len(data[x])), False) if x in train_keys else None,
            shuffle=False,
            num_workers=workers[x]) if y else None for x, y in data_keys.items()}
    else:
        dataloaders = {x: torch.utils.data.DataLoader(
            data[x], batch_size=batch_size[x], shuffle=True if x == "train" else False,
            num_workers=workers[x]) if y else None for x, y in data_keys.items()}
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


def get_dataloader_for_label(dataset_name: str, dataset, label: int,
                             batch_size: int, num_workers: int, shuffle: bool):
    if dataset_name in {"cifar-10", "tiny", "tiny-imagenet-200"}:
        imgs, targets = zip(*[(dataset.data[i], x) for i, x in enumerate(dataset.targets)
                              if x == label])
        data = ImageDataset(imgs, targets, in_memory=True,
                            transform=dataset.transform)
    elif dataset_name == "imagenet":
        imgs, targets = zip(*[img for img in dataset.imgs if img[1] == label])
        data = ImageDataset(imgs, targets, in_memory=False,
                            transform=dataset.transform)
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                         num_workers=num_workers,
                                         shuffle=shuffle, pin_memory=True)

    return loader
