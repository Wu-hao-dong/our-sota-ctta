#!/usr/bin/env bash
set -euo pipefail

# Run from the imagenet/ directory after activating the conda environment:
#   DATA_DIR=/path/to/datasets bash ./bash/ours.sh
# DATA_DIR must be the parent directory containing ImageNet-C/.

: "${DATA_DIR:?Set DATA_DIR to the parent directory containing ImageNet-C/}"
export PYTHONPATH=.

python imagenetc.py \
    --cfg ./cfgs/vit/ours.yaml \
    --data_dir "${DATA_DIR}" \
    "$@"
