# Code for paper "Decomposing the Deep"

- The filters per class for CIFAR-10 dataset are in `resnet20_cifar-10_indices.json`.
  These correspond to the **top 5** filters per class.
- The *influential filters* can vary per instance of the model. We provide the
  filters for CIFAR-10 on the checkpoint given in file `resnet20-12fca82f.th`

# Evaluation

- The code has been tested with `python==3.8` and `torch==1.7.1`, though other
  version may work.
- The recommended method of installation with via `pip`. You can run
  `pip install -r requirements.txt` and then run `python main.py eval`

  First standard resnet will run and then the decomposed final layer with
  indices corresponding to `resnet20_cifar-10_indices.json`.

# Influential Indices Extraction Code

- **Will add soon**. Been a bit busy.
