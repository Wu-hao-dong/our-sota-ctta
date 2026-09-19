#!/usr/bin/env bash
set -euo pipefail

# Run from the cifar/ directory after activating the conda environment:
#   DATA_DIR=/path/to/datasets CHECKPOINT=/path/to/cifar10_source.t7 \
#     bash ./bash/cifar10/ours.sh
# DATA_DIR must be the parent directory containing CIFAR-10-C/.

: "${DATA_DIR:?Set DATA_DIR to the parent directory containing CIFAR-10-C/}"
: "${CHECKPOINT:?Set CHECKPOINT to the CIFAR-10 ViT-B source checkpoint}"
export PYTHONPATH=.

python cifar10c_vit.py \
    --cfg ./cfgs/cifar10/ours.yaml \
    --checkpoint "${CHECKPOINT}" \
    --data_dir "${DATA_DIR}" \
    "$@"
