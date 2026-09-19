export PYTHONPATH=
conda deactivate
conda activate ours

python imagenetc.py --cfg ./cfgs/vit/source.yaml
