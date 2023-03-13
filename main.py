import argparse
import sys
import os
import json
from functools import partial

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from common_pyutil.monitor import Timer

from modified_resnet import get_model
import dataset
from util import have_cuda, load_fixing_names
from indices import Indices


def validate(model, val_loader, gpu=0, print_only=False):
    timer = Timer(True)
    model = model.eval()
    if have_cuda() and gpu is not None:
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    correct = 0
    total = 0
    preds = {"preds": [], "labels": []}
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)
            with timer:
                outputs = model(imgs)
            _preds = F.softmax(outputs, 1).argmax(1)
            correct += torch.sum(_preds == labels)
            preds["preds"].append(_preds.detach().cpu().numpy())
            preds["labels"].append(labels.detach().cpu().numpy())
            total += labels.shape[0]
            if i % 5 == 4:
                print(f"{(i / len(val_loader)) * 100} percent done in {timer.time} seconds")
                print(f"Correct: {correct}, Total: {total}")
                timer.clear()
    if print_only:
        print(f"correct: {correct}, total: {total}, accuracy: {correct/total*100}")
    else:
        return correct, total, preds


def validate_essential_nature(model_name, weights_file, inds_file,
                              data, num_classes, batch_size, gpu=None):
    results = {}
    timer = Timer()
    model = get_model(model_name, weights_file,
                      inds_file=inds_file)
    model = model.eval()
    if gpu is not None:
        model = model.cuda(gpu)
    for x in range(num_classes):
        with timer:
            val_loader = dataset.get_dataloader_for_label(data["name"],
                                                          data["val"],
                                                          x, batch_size, 8, False)
        print(f"Got data for label {x} in {timer.time} seconds")
        with timer:
            model._val_label = x
            outputs = []
            with torch.no_grad():
                for imgs, labels in val_loader:
                    if gpu is not None:
                        imgs = imgs.cuda(gpu)
                    outputs.append(model.forward_noise_at_inds_for_label(imgs))
        results[x] = {}
        results[x]["noise_at_inf"] = sum([(o[0].argmax(1) == x).sum().item() for o in outputs])
        results[x]["noise_at_noninf"] = sum([(o[1].argmax(1) == x).sum().item() for o in outputs])
        print(f"Got result for label {x} in {timer.time} seconds")
    return results


def finetune(model, model_name, dataloaders, num_epochs=10, lr=2e-04, gpu=0):
    if gpu is not None:
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    learning_rate = lr
    criterion = nn.CrossEntropyLoss()
    total_step = len(dataloaders["train"])
    trainable_params = [x for x in model.parameters() if x.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
    # print("Validating small sample for model before training.")
    # validate_subsample(model, dataloaders["val"], gpu)
    timer = Timer()
    epoch_timer = Timer(True)
    loop_timer = Timer()
    total_loss = 0
    for epoch in range(num_epochs):
        model = model.train()
        correct = 0
        total = 0
        with epoch_timer:
            for i, batch in enumerate(dataloaders["train"]):
                with timer:
                    images, labels = batch
                with loop_timer:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    correct += torch.sum(outputs.detach().argmax(1) == labels)
                    total += len(labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                if (i+1) % 10 == 0:
                    print(f"Epoch {epoch+1}, iteration {i+1}/{total_step}," +
                          f" correct {correct}/{total}",
                          f" average loss per batch {total_loss / 10}" +
                          f" in time {loop_timer.time}")
                    total_loss = 0
                    loop_timer.clear()
                i += 1
        print(f"Trained one epoch on device {device} in {epoch_timer.time} seconds")
        print(f"Correct {correct}/{total} for epoch {epoch}")
        print("Validating model")
        correct, total, _ = validate(model, dataloaders["val"], gpu)
        print(f"Correct {correct}/{total}")
        epoch_timer.clear()


def main(model_name, weights_file, inds_file, exp, num_epochs, lr, train_bs, gpu):
    data_name = "cifar-10" if model_name == "resnet20" else "imagenet"
    data, dataloaders, num_classes = dataset.get_data(data_name,
                                                      {"train": train_bs, "val": 128, "test": 128},
                                                      {"train": 12, "val": 12, "test": 12})

    if exp == "essential":
        results_csg = validate_essential_nature("resnet20",
                                                "best_save_CSG.pth",
                                                "resnet20_cifar-10_csg_indices.json", data, 10, 8, 0)
        results_ours = validate_essential_nature("resnet20", "resnet20-12fca82f.th",
                                                 "resnet20_cifar-10_indices.json", data, 10, 8, 0)
        results = {"ours": results_ours, "csg": results_csg}
        with open("results_essential_features.json", "w") as f:
            json.dump(results, f)
    elif exp == "validate":
        model = get_model(model_name, weights_file, inds_file)
        model = model.cuda()
        model = model.eval()
        validate(model, dataloaders["val"], gpu, print_only=True)
    elif exp == "finetune":
        model = get_model(model_name, weights_file, inds_file)
        finetune(model, model_name, dataloaders, num_epochs, lr)
    else:
        raise ValueError(f"Unknown experiment {exp}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("command")
    parser.add_argument("model")
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--inds-file", "-i")
    parser.add_argument("--lr", type=float, default=2e-04)
    parser.add_argument("-b", "--batch-size", type=int, default=64)
    parser.add_argument("--weights-file", "-w")
    parser.add_argument("--gpu", "-g", type=int)
    args = parser.parse_args()
    main(args.model, args.weights_file, args.inds_file, args.command,
         num_epochs=args.num_epochs, lr=args.lr, train_bs=args.batch_size, gpu=args.gpu)
