from benetech_classifier.train import train
import argparse
from types import SimpleNamespace

# default configuration parameters
config = SimpleNamespace(
    data_dir = "/data/bartley/gpu_test/500k_graphs_v2/",    
    model_save_dir = "/data/bartley/gpu_test/models/classifiers/",
    cache_dir = "/data/bartley/gpu_test/HF_CACHE",
    model_path = "B1", # EfficientNet - B1-B4 are implemented
    no_wandb = True,
    project = "Benetech-Classifier",
    save_model = True,
    num_classes = 5,
    batch_size = 16,
    epochs = 1,
    lr = 7e-4,
    lr_min = 1e-8,
    num_cycles = 5,
    label_smoothing = 0.20,
    scheduler = "CosineAnnealingLRDecay",
    transform_type = "center", # center or top_right
    resize_shape = 340,
    val_repeat_n = 25,
    # -- Trainer Config --
    accelerator = "gpu",
    fast_dev_run = False,
    overfit_batches = 0,
    devices = 1,
    precision = 32,
    log_every_n_steps = 50,
    accumulate_grad_batches = 1,
    val_check_interval = 0.01,
    num_workers = 2,
    seed = 0,
    verbose = 2,
)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--val_repeat_n", type=int, default=config.val_repeat_n, help="Number of times to repeat the valiation images (on train all).")
    parser.add_argument("--resize_shape", type=int, default=config.resize_shape, help="Shape to resize images before crop.")
    parser.add_argument("--transform_type", type=str, default=config.transform_type, help="Transformation type to try.")
    parser.add_argument("--scheduler", type=str, default=config.scheduler, help="Learning rate scheduler for the model to use.")
    parser.add_argument("--model_path", type=str, default=config.model_path, help="EfficientNet Model to train.")
    parser.add_argument("--data_dir", type=str, default=config.data_dir, help="Data directory path.")
    parser.add_argument('--train_all', action='store_true', help='Indicator wether to train on all the data.')
    parser.add_argument('--fast_dev_run', action='store_true', help='Check PL modules are set up correctly.')
    parser.add_argument("--overfit_batches", type=int, default=config.overfit_batches, help="Num of batches to overfit (sanity check).")
    parser.add_argument('--no_wandb', action='store_true', help='Wether to log with weights and biases.')
    parser.add_argument("--seed", type=int, default=config.seed, help="Seed for reproducability.")
    parser.add_argument("--batch_size", type=int, default=config.batch_size, help="Num data points per batch.")
    parser.add_argument("--accumulate_grad_batches", type=int, default=config.accumulate_grad_batches, help="Number of steps before each optimizer step.")
    parser.add_argument("--epochs", type=int, default=config.epochs, help="Number of epochs to train.")
    parser.add_argument("--lr", type=float, default=config.lr, help="Starting learning rate for the model.")
    parser.add_argument("--lr_min", type=float, default=config.lr_min, help="Lowest allowed learning rate for the model.")
    parser.add_argument("--num_cycles", type=int, default=config.num_cycles, help="Number of cycles for the cyclical cosine annealing LR.")
    parser.add_argument("--label_smoothing", type=float, default=config.label_smoothing, help="Label smoothing factor for loss function.")
    parser.add_argument("--val_check_interval", type=float, default=config.val_check_interval, help="Number of batches between validation checks.")
    args = parser.parse_args()
    
    # Update config w/ parameters passed through CLI
    for key, value in vars(args).items():
        setattr(config, key, value)

    return config

def main(config):
    module = train(config)
    pass

if __name__ == "__main__":
    config = parse_args()
    main(config)
