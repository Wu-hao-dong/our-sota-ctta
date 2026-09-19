#!/usr/bin/env bash
set -euo pipefail

# Run from the cifar/ directory after activating the conda environment:
#   DATA_DIR=/path/to/datasets CHECKPOINT=/path/to/cifar100_source.t7 \
#     bash ./bash/cifar100/ours.sh
# DATA_DIR must be the parent directory containing CIFAR-100-C/.

: "${DATA_DIR:?Set DATA_DIR to the parent directory containing CIFAR-100-C/}"
: "${CHECKPOINT:?Set CHECKPOINT to the CIFAR-100 ViT-B source checkpoint}"
export PYTHONPATH=.

python cifar100c_vit.py \
    --cfg ./cfgs/cifar100/ours.yaml \
    --checkpoint "${CHECKPOINT}" \
    --data_dir "${DATA_DIR}" \
    "$@"
