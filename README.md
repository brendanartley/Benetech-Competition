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
CUDA_VISIBLE_DEVICES=0 python main.py --seed=0 --batch_size=3 --accumulate_grad_batches=5 --fast_dev_run
CUDA_VISIBLE_DEVICES=1 python main.py --batch_size=3
CUDA_VISIBLE_DEVICES=2 python old_validate.py
CUDA_VISIBLE_DEVICES=0 python old_validate.py
```