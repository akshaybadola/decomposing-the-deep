# README

- This is the code for the paper [Decomposing the Deep](https://arxiv.org/abs/2112.07719). The model URLs we had
  experimented with are given below in python dict format. They're also present  in `modified_resnet.py`.
- To conduct detailed experiments, these model weights were used.
- You'll need pytorch and preferably a GPU.
- The python requirements are given in requirements.txt.
- You'll need the CIFAR-10 data present in a `data` folder to run the experiments
- The code contains additional in-progress/incomplete experiments and methods
   and may not run properly for all commands.
- The *influential filters* can vary according to the trained instance of the model.
  We provide the filters for CIFAR-10 on the checkpoint given in file `resnet20-12fca82f.th`

### Model URLs

    model_urls = {
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
        'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
        'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
        'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
        'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
    }

## Evaluation

- The code has been tested with `python==3.8` and `torch==1.7.1`, though other
  version may work. See `requirements.txt`
- The recommended method of installation with via `pip`. You can run
  `pip install -r requirements.txt`.

### Validation with decomposed final layer

The resnet20 model with CIFAR-10 can be finetuned with:
`python main.py validate resnet20 -i resnet20_cifar-10_indices.json -w resnet20-12fca82f.th --gpu 0`
`gpu` is optional for validation. This only shows the accuracy on the validation set for the
decomposed layer, which is `88.59%`.

### Validation on the original pre-trained model
`python main.py validate resnet20 -w resnet20-12fca82f.th --gpu 0`

Same as previous command but indices are not given. Shows accuracy on the validation set for the original
trained model without decomposition which is `91.71%`.

### Validation with gaussian noise for quantifying essentiality

You can run `python main.py essential resnet20 -w resnet20-12fca82f.th --gpu 0` to run the validation code for running Resnet20 on CIFAR-10 with Gaussian Noise. This saves the results to `results_essential_features.json` for both our method and CSG.

### Mutual information calculation

Mutual information score can be calculated by running `python mutual_info.py`. Shows for both
our method and CSG.

## Influential Indices Extraction Code

- **Will add soon** in case someone asks. The code exists in `l1_norm.py` I think but there
  are multiple files scattered across various folders and I haven't consolidated them.

## Imagenet and Resnet50

Commands for resnet50 and imagenet are similar except model name and dataset name are changed.

E.g.: `python main.py validate resnet50 -i resnet50_imagenet_indices.json --gpu 0 --lr 0.00001`
Lower `lr` in general for Imagenet and Resnet50
Weights are loaded from model_urls.

## Fine-tuning

The resnet20 model with CIFAR-10 can be finetuned with:
`python main.py finetune resnet20 -i resnet20_cifar-10_indices.json -w resnet20-12fca82f.th --gpu 0 --lr 0.0001`

GPU is strongly recommended. Accuracy of around `91.30`--`91.40` after first epoch depending on random seed.
Additional epochs increase accuracy but it does saturate or overfit after a few epochs.
We have observed accuracies upto `91.50`--`91.60` which is statistically insignificant from the original.

The resnet50 model with Imagenet can be finetuned with:
`python main.py finetune resnet50 -i resnet50_imagenet_indices.json --gpu 0 --lr 0.000001`

## Running with Imagenet
You need to have a folders `imagenet/ILSVRC/{train,val}` beneath `data`.

Simply substitute `resnet50` for model name and `imagenet` for dataset for the experiments.
