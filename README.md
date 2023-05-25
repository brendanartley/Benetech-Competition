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
# -------- Decoder Stuff ---------
# Training scripts
CUDA_VISIBLE_DEVICES=0 python main.py --seed=0 --epochs=2 --val_check_interval=0.26
CUDA_VISIBLE_DEVICES=1 python main.py --seed=1 --epochs=2
CUDA_VISIBLE_DEVICES=1 python main.py --epochs=1000 --overfit_batches=1 --lr=0.00003
CUDA_VISIBLE_DEVICES=2 python main.py --epochs=10 --model_path

CUDA_VISIBLE_DEVICES=1 python main.py --seed=1 --epochs=10
CUDA_VISIBLE_DEVICES=2 python main.py --seed=1 --epochs=10 --processor_path="google/pix2struct-base" --model_path="google/pix2struct-base"
CUDA_VISIBLE_DEVICES=3 python main.py --seed=1 --epochs=5 --val_check_interval=0.26

# Current evaluation scripts
CUDA_VISIBLE_DEVICES=0 python infer_image.py
CUDA_VISIBLE_DEVICES=1 python infer_image.py
CUDA_VISIBLE_DEVICES=2 python infer_image.py

# ---------- Classifier Stuff ----------
CUDA_VISIBLE_DEVICES=1 python classifier_train.py --batch_size=128 --lr=0.00001 --val_check_interval=0.10
CUDA_VISIBLE_DEVICES=2 python classifier_train.py --batch_size=128 --lr=0.0001 --val_check_interval=0.10

# ---------- Sweeps ----------
CUDA_VISIBLE_DEVICES=0 wandb agent brendanartley/Benetech-Classifier/82uw2f7z
CUDA_VISIBLE_DEVICES=1 wandb agent brendanartley/Benetech-Classifier/82uw2f7z
CUDA_VISIBLE_DEVICES=2 wandb agent brendanartley/Benetech-Classifier/82uw2f7z
CUDA_VISIBLE_DEVICES=3 wandb agent brendanartley/Benetech-Classifier/82uw2f7z
```