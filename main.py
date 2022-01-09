import sys
import os
from functools import partial

import torch
import torchvision
import torch.nn.functional as F
from common_pyutil.monitor import Timer

import modified_resnet
import dataset
from util import have_cuda, load_fixing_names


def load_stuff(model_name, data_name, weights_file):
    bs = 128
    no_transform = True
    data, dataloaders, num_classes = dataset.get_data(data_name, {"train": bs, "val": bs},
                                                      {"train": 12, "val": 12},
                                                      no_transform)
    if data_name == "imagenet":
        pretrained = True
    else:
        pretrained = False
    if any([x in model_name for x in ["densenet", "efficientnet"]]):
        model = getattr(torchvision.models, model_name)(pretrained=pretrained)
    elif "resnet" in model_name:
        model = modified_resnet.get_model(weights_file)
        # NOTE: resnet.py not given right now
        # model = getattr(resnet, model_name)(num_classes, pretrained)
    else:
        raise ValueError(f"Unsupported model {model_name}")
    if data_name != "imagenet":
        ckpt = torch.load(weights_file, map_location="cpu")
        if "state_dict" in ckpt:
            status = load_fixing_names(model, ckpt["state_dict"])
        elif "model_state_dict" in ckpt:
            status = load_fixing_names(model, ckpt["model_state_dict"])
        print(status)
    else:
        print(f"Loaded pretrained model {model_name}")
    model = model.eval()
    model.name = model_name
    return model, data, dataloaders, num_classes


def validate(model, val_loader, gpu=0):
    timer = Timer(True)
    model.eval()
    if have_cuda():
        model = model.cuda(gpu)
    correct = 0
    total = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            imgs, labels = batch
            if have_cuda():
                imgs, labels = imgs.cuda(gpu), labels.cuda(gpu)
            with timer:
                outputs = model.forward(imgs)
            correct += torch.sum(F.softmax(outputs, 1).argmax(1) == labels)
            total += labels.shape[0]
            if i % 5 == 4:
                print(f"{(i / len(val_loader)) * 100} percent done in {timer.time} seconds")
                timer.clear()
    return correct, total


def main():
    if sys.argv[1] == "eval":
        model_name = "resnet20"
        data_name = "cifar-10"
        model, data, dataloaders, num_classes = load_stuff(
            model_name, data_name, "resnet20-12fca82f.th")
        # Standard evaluation
        model.forward = model.forward_regular
        print("Computing for Regular Pretrained Resnet")
        print("Result for regular pretrained resnet: ",
              validate(model, dataloaders["val"]))
        # Decomposed resnet
        model.forward = model.forward_decomposed
        print("\nComputing for Decomposed Resnet")
        print("Result for decomposed resnet: ",
              validate(model, dataloaders["val"]))
    else:
        raise NotImplementedError("Rest of the things not here yet")


if __name__ == '__main__':
    main()
