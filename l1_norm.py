from typing import Tuple
from functools import partial
import pickle

import torch
from torch.nn import functional as F

from common_pyutil.monitor import Timer

from util import have_cuda
import dataset


def subroutine_resnet(model, batch, gpu=None):
    with torch.no_grad():
        if gpu is not None:
            x = model.head(batch.cuda(gpu))
        else:
            x = model.head(batch)
        return F.avg_pool2d(x, x.shape[-1]).view(x.shape[0], -1)


def subroutine_densenet(model, batch, gpu=None):
    with torch.no_grad():
        if gpu is not None:
            x = model.features(batch.cuda(gpu))
        else:
            x = model.features(batch)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return torch.flatten(x, 1)


def subroutine_efficientnet(model, batch, gpu=None):
    with torch.no_grad():
        if gpu is not None:
            x = model.avgpool(model.features(batch.cuda(gpu)))
        else:
            x = model.avgpool(model.features(batch))
        return torch.flatten(x, 1)


def final_weights(model, model_name) -> Tuple[torch.Tensor, torch.Tensor]:
    if "resnet" in model_name:
        final = getattr(model, "fc", None) or getattr(model, "linear")
    elif "densenet" in model_name:
        final = getattr(model, "fc", None) or getattr(model, "classifier")
    elif "efficientnet" in model_name:
        final = getattr(model, "classifier")[1]
    else:
        raise ValueError("Only resnet and densenet models are allowed")
    return final.weight, final.bias


def get_indices(models, data_name, data, num_filters=[], gpus=None):
    if isinstance(num_filters, int):
        num_filters = [num_filters]
    if data_name == "imagenet":
        classes = 1000
        if not num_filters:
            num_filters = (10, 100+10, 10)
    elif "tiny" in data_name.lower():
        classes = 200
        if not num_filters:
            num_filters = (5, 50+5, 5)
    elif data_name == "cifar-10":
        classes = 10
        if not num_filters:
            num_filters = (5, 30+5, 5)
    else:
        raise ValueError("Unknown data given")
    funcs = {}
    if not isinstance(models, list):
        models = [models]
    if not isinstance(gpus, list):
        gpus = [None for _ in models]
    for i, model in enumerate(models):
        model_name = model.name.lower()
        if "resnet" in model_name:
            model_type = "resnet"
        elif "densenet" in model_name:
            model_type = "densenet"
        elif "efficientnet" in model_name:
            model_type = "efficientnet"
        else:
            raise ValueError("Only resnet and densenet variants are supported for now")
        funcs[model_name] = partial(top_filters_for_class,
                                    model, model_type, gpu=gpus[i])
    result = {}
    timer = Timer()
    for c in range(classes):
        print(f"Running for {(c+1)} out of {classes} classes")
        with timer:
            img_batches = dataset.get_data_for_label(data_name, data, c, 128)
        print(f"Got batches in {timer.time} seconds")
        with timer:
            for model_name in funcs:
                if model_name not in result:
                    result[model_name] = {}
                func = funcs[model_name]
                if c not in result[model_name]:
                    result[model_name][c] = {}
                for k in num_filters:
                    result[model_name][c][k] = func(img_batches, k=k)
        print(f"Got result for models {funcs.keys()} and class {c} in {timer.time} seconds")
    with open("testtest.pkl", "wb") as f:
        pickle.dump(result, f)
    return result


def top_filters_for_class(model, model_name, data, k, gpu):
    model.eval()
    if have_cuda() and gpu is not None:
        model = model.cuda(gpu)
    else:
        gpu = None
    if "resnet" in model_name.lower():
        subr = subroutine_resnet
    elif "densenet" in model_name.lower():
        subr = subroutine_densenet
    elif "efficientnet" in model_name.lower():
        subr = subroutine_efficientnet
    out = []
    with torch.no_grad():
        for batch in data:
            responses = subr(model, batch, gpu)
            norm_resp = (responses.T / responses.sum(1)).T
            norm_resp, inds_resp = norm_resp.sort(1, descending=True)
            mass = norm_resp[:, :k].sum(1)
            out.append({"inds": inds_resp[:, :k].cpu().numpy(),
                        "mass": mass})
    return out
