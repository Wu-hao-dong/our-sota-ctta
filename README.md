# Ours

Official-style research code for `ours`, a boundary-free continual
test-time adaptation (CTTA) method evaluated on ImageNet-C, CIFAR-10-C, and CIFAR-100-C.

This repository is organized from the official [ViDA](https://github.com/Yangsenqiao/vida)
codebase. It retains the Source, Tent, and CoTTA baselines and adds the released method with
the smallest possible changes to the original evaluation structure.

## Method overview

The source ViT backbone is frozen during online optimization. Each fused attention `qkv` layer is
augmented with shared and private low-rank branches that affect only query and value:

- an online covariance-fingerprint detector discovers domain changes without corruption labels;
- at a detected boundary, one label-free entropy gradient is split into a history-aligned shared
  subspace and a history-residual private subspace;
- only the low-rank up-projections are optimized online;
- completed updates are consolidated into the frozen backbone before fresh subspaces are created;
- the released update uses confidence-filtered, within-batch symmetric sharpness-aware adaptation.

The public adaptation identifier is:

```yaml
MODEL:
  ADAPTATION: ours
```

## Repository layout

```text
.
├── cifar/                         # CIFAR-10-C and CIFAR-100-C
│   ├── bash/                      # launch scripts
│   ├── cfgs/                      # dataset and method configurations
│   ├── cifar10c_vit.py
│   ├── cifar100c_vit.py
│   ├── inject_ours.py
│   └── ours.py
├── imagenet/                      # ImageNet-C
│   ├── bash/
│   ├── cfgs/
│   ├── imagenetc.py
│   ├── inject_ours.py
│   └── ours.py
├── environment.yml
└── LICENSE
```

The CIFAR and ImageNet trees are intentionally self-contained, following the upstream ViDA
layout. Method changes therefore need to be mirrored in both trees.

## Installation

The released Conda environment is named `ours` and uses package versions validated in WSL
(Python 3.9.25, PyTorch 2.8.0+cu128, torchvision 0.23.0, timm 1.0.26):

```bash
conda env create -f environment.yml
conda activate ours
```

Experiments require an NVIDIA GPU. Run each script from its task directory so the vendored
`robustbench` package is resolved correctly.

## Data and source checkpoints

The launch scripts expect `DATA_DIR` to point to a parent directory with this layout:

```text
DATA_DIR/
├── ImageNet-C/
├── CIFAR-10-C/
└── CIFAR-100-C/
```

- ImageNet-C: download from the [official ImageNet-C archive](https://zenodo.org/records/2235448).
  The ImageNet experiment uses timm's pretrained `vit_base_patch16_224` source model and does not
  require a method-specific checkpoint.
- CIFAR-10 source ViT-B checkpoint: use the
  [checkpoint linked by ViDA](https://drive.google.com/file/d/1pAoz4Wwos74DjWPQ5d-6ntyjQkmp9FPE/view?usp=sharing).
- CIFAR-100 source ViT-B checkpoint: use the
  [checkpoint linked by ViDA](https://drive.google.com/file/d/1yRekkpkIdwX_LFsOh4Ba9ndaECnY-UC-/view?usp=sharing).

Datasets, checkpoints, and generated outputs are deliberately excluded from this repository.

## Running the released method

### ImageNet-C

```bash
cd imagenet
DATA_DIR=/path/to/datasets \
  bash ./bash/ours.sh
```

### CIFAR-10-C

```bash
cd cifar
DATA_DIR=/path/to/datasets \
CHECKPOINT=/path/to/cifar10_source.t7 \
  bash ./bash/cifar10/ours.sh
```

### CIFAR-100-C

```bash
cd cifar
DATA_DIR=/path/to/datasets \
CHECKPOINT=/path/to/cifar100_source.t7 \
  bash ./bash/cifar100/ours.sh
```

The scripts accept additional YACS overrides after the command. For example, an ImageNet-C
long-stream control can be launched with:

```bash
DATA_DIR=/path/to/datasets \
  bash ./bash/ours.sh CORRUPTION.ROUNDS 10
```

## Baselines

The Source, Tent, and CoTTA methods and launch scripts are retained for direct comparison. Their
checkpoint placeholders and commands follow the upstream repository.

## Acknowledgements

This code builds on and vendors components from:

- [ViDA](https://github.com/Yangsenqiao/vida)
- [CoTTA](https://github.com/qinenergy/cotta)
- [RobustBench](https://github.com/RobustBench/robustbench)
- [KATANA](https://github.com/giladcohen/KATANA)

Please cite the corresponding upstream works when using their code or ideas. The original MIT
license and vendored third-party license files are preserved.

## License

Released under the MIT License. See [LICENSE](LICENSE).
