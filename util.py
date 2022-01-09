import numpy as np
import torch


def have_cuda():
    return torch.cuda.is_available()


def json_defaults(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    elif isinstance(x, torch.Tensor):
        return x.cpu().numpy().tolist()
    else:
        return x

def load_fixing_names(model, state_dict):
    model_keys = [*model.state_dict().keys()]
    keys = [*state_dict.keys()]
    for k in keys:
        if k in model_keys:
            continue
        if k.replace("module.", "") in model_keys:
            state_dict[k.replace("module.", "")] = state_dict.pop(k)
    return model.load_state_dict(state_dict, strict=False)
