import lightning.pytorch as pl
from benetech_encoder.modules import BenetechDataModule, BenetechModule
from benetech_encoder.helpers import load_logger_and_callbacks

def train(
        config,
):
    pl.seed_everything(config.seed, workers=True)

    if config.fast_dev_run == True:
        config.num_workers = 1

    data_module = BenetechDataModule(
        data_dir = config.data_dir,
        val_repeat_n = config.val_repeat_n,
        batch_size = config.batch_size,
        processor_path = config.processor_path,
        max_length = config.max_length,
        num_workers = config.num_workers,
        max_patches = config.max_patches,
        cache_dir = config.cache_dir,
        train_all = config.train_all,
        chart_type = config.chart_type,
        axis = config.axis,
    )

    logger, callbacks = load_logger_and_callbacks(
        fast_dev_run = config.fast_dev_run,
        metrics = {"val_loss": "min", "train_loss": "min"},
        overfit_batches = config.overfit_batches,
        no_wandb = config.no_wandb,
        project = config.project,
    )

    module = BenetechModule(
        lr = config.lr,
        lr_min = config.lr_min,
        max_length = config.max_length,
        model_path = config.model_path,
        model_save_dir = config.model_save_dir,
        processor = data_module.processor,
        run_name = logger._experiment.name if logger else None,
        save_model = config.save_model,
        cache_dir = config.cache_dir,
        scheduler = config.scheduler,
        chart_type = config.chart_type,
        axis = config.axis,
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
        # strategy= "ddp" if config.devices > 1 else None,
        callbacks = callbacks,
        logger = logger,
        log_every_n_steps = config.log_every_n_steps,
        accumulate_grad_batches = config.accumulate_grad_batches,
        val_check_interval = config.val_check_interval,
        enable_checkpointing = False,
        gradient_clip_val = 1.0,
    )

    trainer.fit(module, datamodule=data_module)

    # TODO: Set up code to run generation and score with benetech_score() method
    # trainer.test(module, datamodule=data_module)
    return