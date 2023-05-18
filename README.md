## Benetech

Files for the Benetech - Making Graphs Accessible competition.

## GPU Notes

22,919 MB per node

GPU info: `watch nvidia-smi`

Watch log file: `tail -n 25 -f logfile`

## TODO

Set up evaluation script using trainer.test()
- Log preds to txt
- Score w/ benetech

Set up a cache directory in /data
- https://huggingface.co/docs/datasets/cache#cache-directory

## Ideas

One model per chart type. Inference w/ fp16? (will it fit in memory?)

## Sweeps Notes

Wandb Sweeps Docs: https://docs.wandb.ai/guides/sweeps
Sweeps Lesson Files: https://github.com/wandb/edu/tree/main/mlops-001/lesson2

Run a sweep on a specific GPU
```
# Training scripts
CUDA_VISIBLE_DEVICES=0 python main.py
CUDA_VISIBLE_DEVICES=1 python main.py --seed=1

# Current evaluation scripts
CUDA_VISIBLE_DEVICES=2 python old_validate.py
CUDA_VISIBLE_DEVICES=0 python old_validate.py
CUDA_VISIBLE_DEVICES=1 python test.py
```