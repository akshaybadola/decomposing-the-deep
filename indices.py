from pathlib import Path
import pickle
import json

import numpy as np


class Indices:
    def __init__(self, file=None):
        self._inds = {}
        if file:
            try:
                with open(file, "rb") as f:
                    inds = pickle.load(f)
                vals = [*inds.values()]
                if vals and not isinstance(vals[0], np.ndarray):
                    classes = [*[*inds.values()][0].keys()]
                    inds = {k: np.array([inds[k][i] for i in classes]) for k in inds}
                self._inds = inds
            except pickle.UnpicklingError:
                try:
                    with open(file) as f:
                        inds = json.load(f)
                    inds = {int(k): v for k, v in inds.items()}
                    if len(inds) == 10:
                        classes = [*range(10)]
                        self._inds = {(3, 2, 2): np.array([inds[i] for i in classes])}
                    else:
                        classes = [*range(1000)]
                        self._inds = {(4, 2, 3): np.array([inds[i] for i in classes])}
                except Exception:
                    print("Could not load file")
                    self._inds = {}
        else:
            self._inds = {}
        self.keys = [*self._inds.keys()]

    def __repr__(self):
        return self.inds.__repr__()

    def __len__(self):
        return len(self._inds)

    def load(self, file):
        with open(file, "rb") as f:
            inds = pickle.load(f)
            classes = [*[*inds.values()][0].keys()]
            inds = {k: np.array([inds[k][i] for i in classes]) for k in inds}
        self._inds = inds
        self.keys = [*self._inds.keys()]

    def dump(self, file, overwrite=False):
        if Path(file).exists() and not overwrite:
            print("File exists. Set overwrite to true to overwrite")
        else:
            with open(file, "wb") as f:
                pickle.dump(self._inds, f)

    def add(self, key, inds):
        """Add indices at key"""
        if isinstance(inds, dict):
            classes = [int(x) for x in inds.keys()]
            classes.sort()
            inds = np.array([np.array(inds[k]) for k in classes])
        self._inds[key] = inds
        self.keys = [*self._inds.keys()]

    @property
    def inds(self):
        return self._inds

    @inds.setter
    def inds(self, x):
        self._inds = x

    @property
    def keys(self):
        return self._keys

    @keys.setter
    def keys(self, x):
        self._keys = x
        self.keys.sort()

    def __getitem__(self, x):
        if isinstance(x, int):
            key = tuple([*self.keys][x])
        elif iter(x) and x in self.keys:
            key = x
        elif any([i < 0 for i in x]):
            keys = [*map(lambda x: list(set(x)), zip(*self.keys))]
            for k in keys:
                k.sort()
            key = tuple([*map(lambda x, y: x[y], self.keys, x)])
        return self.inds[key]
