import json

import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score

import torch


with open("resnet20_cifar-10_indices.json") as f:
    inds_5 = json.load(f)

with open("resnet20_cifar-10_10_10_indices.json") as f:
    inds_10 = json.load(f)

with open("resnet20_cifar-10_20_20_indices.json") as f:
    inds_20 = json.load(f)

inds = {5: inds_5, 10: inds_10, 20: inds_20}
our_pairwise_mi_score = {}
for k in [5, 10, 20]:
    our_pairwise_mi_score[k] = []
    for a in inds[k].values():
        for b in inds[k].values():
            ar_a = np.zeros_like(range(64))
            ar_b = np.zeros_like(range(64))
            ar_a[a] = 1
            ar_b[b] = 1
            our_pairwise_mi_score[k].append(mutual_info_score(ar_a, ar_b))
    our_pairwise_mi_score[k] = np.mean(our_pairwise_mi_score[k])
    # print(np.mean(our_pairwise_mi_score[k]))
    # 0.03453775959861226 @ k == 5

csg_weights = torch.load("best_save_CSG.pth")
csg = csg_weights["model_state_dict"]["csg"].cpu().numpy()


csg_pairwise_mi_score = {}
for k in [5, 10, 20]:
    csg_pairwise_mi_score[k] = []
    csg_inds = csg.argsort(1)[:, :k]
    for a in csg_inds:
        for b in csg_inds:
            ar_a = np.zeros_like(range(64))
            ar_b = np.zeros_like(range(64))
            ar_a[a] = 1
            ar_b[b] = 1
            csg_pairwise_mi_score[k].append(mutual_info_score(ar_a, ar_b))
    csg_pairwise_mi_score[k] = np.mean(csg_pairwise_mi_score[k])
    # print(np.mean(csg_pairwise_mi_score[k]))
    # 0.03750908408724534 @ k == 5

df = pd.DataFrame.from_dict({"Ours": our_pairwise_mi_score,
                             "CSG": csg_pairwise_mi_score})
print(df.to_markdown(tablefmt="grid"))
