import lightning.pytorch as pl

def load_logger_and_callbacks(
    fast_dev_run,
    metrics,
    overfit_batches,
    no_wandb,
    project,
):
    """
    Function that loads logger and callbacks.
    
    Returns:
        logger: lighting logger
        callbacks: lightning callbacks
    """
    # Params used to check for Bugs/Errors in Implementation
    if fast_dev_run or overfit_batches > 0 or no_wandb == True:
        logger, callbacks = None, None
    else:
        logger, id_ = get_logger(
            metrics = metrics, 
            project = project, 
            )
        callbacks = get_callbacks()
    return logger, callbacks

def get_logger(metrics, project):
    """
    Function to load logger.
    
    Returns:
        logger: lighting logger
        id_: experiment id
    """
    logger = pl.loggers.WandbLogger(
        project = project, 
        save_dir = None,
        )
    id_ = logger.experiment.id
    
    for metric, summary in metrics.items():
        logger.experiment.define_metric(metric, summary=summary)
    
    return logger, id_

def get_callbacks():
    callbacks = [
        pl.callbacks.LearningRateMonitor(),
    ]
    return callbacks