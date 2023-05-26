import pytorch_lightning as pl
from benetech_classifier.modules import BenetechClassifierModule, BenetechClassifierDataModule
from benetech_classifier.helpers import load_logger_and_callbacks

def train(
        config,
):
    pl.seed_everything(config.seed, workers=True)

    if config.fast_dev_run == True:
        config.num_workers = 1

    data_module = BenetechClassifierDataModule(
        data_dir = config.data_dir,
        batch_size = config.batch_size,
        num_workers = config.num_workers,
        cache_dir = config.cache_dir,
        model_path = config.model_path,
    )

    logger, callbacks = load_logger_and_callbacks(
        fast_dev_run = config.fast_dev_run,
        metrics = {
            "val_loss": "min", 
            "train_loss": "min",
            "val_acc": "max",
            "train_acc": "max",
            },
        overfit_batches = config.overfit_batches,
        no_wandb = config.no_wandb,
        project = config.project,
    )

    module = BenetechClassifierModule(
        learning_rate = config.lr,
        model_save_dir = config.model_save_dir,
        model_path = config.model_path,
        run_name = logger._experiment.name if logger else None,
        save_model = config.save_model,
        cache_dir = config.cache_dir,
        num_classes = config.num_classes,
        label_smoothing = config.label_smoothing,
        epochs = config.epochs,
        scheduler = config.scheduler,
    )

    # Trainer Args: https://lightning.ai/docs/pytorch/stable/common/trainer.html#benchmark
    trainer = pl.Trainer(
        accelerator = config.accelerator,
        benchmark = True, # set to True if input size does not change (increases speed)
        devices = config.devices,
        fast_dev_run = config.fast_dev_run,
        max_epochs = config.epochs,
        num_sanity_val_steps = 1,
        overfit_batches = config.overfit_batches,
        precision = config.precision,
        callbacks = callbacks,
        logger = logger,
        log_every_n_steps = config.log_every_n_steps,
        accumulate_grad_batches = config.accumulate_grad_batches,
        val_check_interval = config.val_check_interval,
        enable_checkpointing = False,
        gradient_clip_val = 1.0,
    )

    trainer.fit(module, datamodule=data_module)
    return