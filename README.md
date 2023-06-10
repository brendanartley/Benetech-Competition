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

## TransCrowd

Code: https://github.com/dk-liang/TransCrowd
Paper: https://arxiv.org/pdf/2104.09116.pdf

## Sweeps Notes

Wandb Sweeps Docs: https://docs.wandb.ai/guides/sweeps
Sweeps Lesson Files: https://github.com/wandb/edu/tree/main/mlops-001/lesson2

## Object Detection (Maybe)

HF Finetuning
https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/YOLOS/Fine_tuning_YOLOS_for_object_detection_on_custom_dataset_(balloon).ipynb#scrollTo=9r-lMAWKWoLY

Run a sweep on a specific GPU

cd /home/bartley/gpu_test/Benetech
```
# -------- Decoder Stuff ---------
# Training scripts
CUDA_VISIBLE_DEVICES=0 python main.py --epochs=1 --axis="x" --chart_type="v" --train_all && CUDA_VISIBLE_DEVICES=0 python main.py --epochs=1 --axis="y" --chart_type="l" --train_all
CUDA_VISIBLE_DEVICES=1 python main.py --epochs=1 --axis="x" --chart_type="h" --train_all && CUDA_VISIBLE_DEVICES=1 python main.py --epochs=1 --axis="y" --chart_type="s" --train_all
CUDA_VISIBLE_DEVICES=2 python main.py --epochs=1 --axis="x" --chart_type="l" --train_all && CUDA_VISIBLE_DEVICES=2 python main.py --epochs=1 --axis="y" --chart_type="v" --train_all
CUDA_VISIBLE_DEVICES=3 python main.py --epochs=1 --axis="x" --chart_type="s" --train_all && CUDA_VISIBLE_DEVICES=3 python main.py --epochs=1 --axis="y" --chart_type="h" --train_all

CUDA_VISIBLE_DEVICES=3 python main.py --epochs=1 --axis="x" --chart_type="d" --train_all && CUDA_VISIBLE_DEVICES=3 python main.py --epochs=1 --axis="y" --chart_type="d" --train_all

CUDA_VISIBLE_DEVICES=0 python main.py --epochs=1 --axis="x" --chart_type="v" --fast_dev_run
CUDA_VISIBLE_DEVICES=1 python main.py --epochs=1 --val_check_interval=0.05 --axis="y" --chart_type="v"
CUDA_VISIBLE_DEVICES=2 python main.py --epochs=1 --val_check_interval=0.05 --axis="x" --chart_type="l"
CUDA_VISIBLE_DEVICES=3 python main.py --epochs=1 --val_check_interval=0.05 --axis="y" --chart_type="l"

# ---------- Counter Stuff ----------
CUDA_VISIBLE_DEVICES=0 python counter_train.py --chart_type="v" --fast_dev_run
CUDA_VISIBLE_DEVICES=1 python counter_train.py --chart_type="h" --fast_dev_run
CUDA_VISIBLE_DEVICES=2 python counter_train.py --chart_type="s" --fast_dev_run
CUDA_VISIBLE_DEVICES=3 python counter_train.py --chart_type="l" --fast_dev_run

# ---------- Classifier Stuff ----------
CUDA_VISIBLE_DEVICES=0 python classifier_train.py --seed=0 --train_all
CUDA_VISIBLE_DEVICES=1 python classifier_train.py --seed=1 --train_all
CUDA_VISIBLE_DEVICES=2 python classifier_train.py --seed=2 --train_all
CUDA_VISIBLE_DEVICES=3 python classifier_train.py --seed=3 --train_all
CUDA_VISIBLE_DEVICES=3 python classifier_train.py --seed=4 --train_all
CUDA_VISIBLE_DEVICES=0 python classifier_train.py --no_wandb --fast_dev_run

# ---------- Sweeps ----------
CUDA_VISIBLE_DEVICES=0 wandb agent brendanartley/Benetech-Classifier/5i8uzcfa
CUDA_VISIBLE_DEVICES=1 wandb agent brendanartley/Benetech-Classifier/5i8uzcfa
CUDA_VISIBLE_DEVICES=2 wandb agent brendanartley/Benetech-Classifier/5i8uzcfa
CUDA_VISIBLE_DEVICES=3 wandb agent brendanartley/Benetech-Classifier/5i8uzcfa
```