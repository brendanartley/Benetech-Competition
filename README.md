## Benetech

Files for the Benetech - Making Graphs Accessible competition.

## GPU Notes

22,919 MB per node

GPU info: `watch nvidia-smi`

Watch log file: `tail -n 25 -f logfile`

## Saved Weights

Downloading weights file
`cp PATH_TO_WEIGHTS ./`


## Sweeps Notes

Wandb Sweeps Docs: https://docs.wandb.ai/guides/sweeps
Sweeps Lesson Files: https://github.com/wandb/edu/tree/main/mlops-001/lesson2

Run a sweep on a specific GPU
```
CUDA_VISIBLE_DEVICES=0 wandb agent sweep_ID
CUDA_VISIBLE_DEVICES=1 wandb agent sweep_ID

CUDA_VISIBLE_DEVICES=0 wandb agent brendanartley/GISLR-keras/tk6ltw1z
CUDA_VISIBLE_DEVICES=1 wandb agent brendanartley/GISLR-keras/tk6ltw1z
CUDA_VISIBLE_DEVICES=2 wandb agent brendanartley/GISLR-keras/tk6ltw1z
```

## Training Notes

Test Run
```
CUDA_VISIBLE_DEVICES=0 python train.py --max_epochs=100 --file="gislr-mw-24"
CUDA_VISIBLE_DEVICES=1 python train.py --max_epochs=100 --file="gislr-mw-24"
CUDA_VISIBLE_DEVICES=2 python train.py --max_epochs=100 --file="gislr-mw-24"
```

## Testing Donut

CUDA_VISIBLE_DEVICES=0,1,2 python train.py --config config/train_BARTLEY.yaml \
    --exp_version "test_experiment"