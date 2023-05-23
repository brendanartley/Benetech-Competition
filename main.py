from benetech_decoder.train import train
import argparse
from types import SimpleNamespace

# defaults
config = SimpleNamespace(
    data_dir = "/data/bartley/gpu_test/bartley-benetech-resized/",    
    # data_dir = "/data/bartley/gpu_test/bartley-benetech-resized-2/",    
    # data_dir = "/data/bartley/gpu_test/bartley-benetech-resized-small/",
    model_save_dir = "/data/bartley/gpu_test/models/",
    # model_path = "google/pix2struct-base",
    # processor_path = "google/pix2struct-base",
    model_path = "google/deplot",
    processor_path = "google/deplot",
    no_wandb = False,
    project = "Benetech",
    save_model = True,
    max_patches = 1024,
    max_length = 512,
    batch_size = 4,
    epochs = 1,
    lr = 3e-5,
    verbose = 2,
    num_workers = 8,
    seed = 0,
    # -- Trainer Config --
    accelerator = "gpu",
    fast_dev_run = False,
    overfit_batches = 0,
    devices = 1,
    precision = 32,
    log_every_n_steps = 100,
    accumulate_grad_batches = 1,
    val_check_interval = None,
)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str, default=config.data_dir, help="Data directory path.")
    parser.add_argument("--model_path", type=str, default=config.model_path, help="Huggingface model path.")
    parser.add_argument("--processor_path", type=str, default=config.processor_path, help="Huggingface processor path.")
    parser.add_argument('--fast_dev_run', action='store_true', help='Check PL modules are set up correctly.')
    parser.add_argument("--overfit_batches", type=int, default=config.overfit_batches, help="Num of batches to overfit (sanity check).")
    parser.add_argument('--no_wandb', action='store_true', help='Wether to log with weights and biases.')
    parser.add_argument("--seed", type=int, default=config.seed, help="Seed for reproducability.")
    parser.add_argument("--batch_size", type=int, default=config.batch_size, help="Num data points per batch.")
    parser.add_argument("--accumulate_grad_batches", type=int, default=config.accumulate_grad_batches, help="Number of steps before each optimizer step.")
    parser.add_argument("--max_patches", type=int, default=config.batch_size, help="Max patches for the pix2sctruct model.")
    parser.add_argument("--epochs", type=int, default=config.epochs, help="Number of epochs to train.")
    parser.add_argument("--lr", type=float, default=config.lr, help="Learning rate for the model.")
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
