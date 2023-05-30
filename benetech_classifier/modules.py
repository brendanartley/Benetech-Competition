import lightning.pytorch as pl
import torchvision
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import torchinfo
import torch.nn.functional as F

from timm.scheduler.cosine_lr import CosineLRScheduler

import pandas as pd
from PIL import Image

class BenetechClassifierDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, train=True, transform=None, train_all=False):
        self.data_dir = data_dir
        self.imgs, self.labels = self.load_df(train, train_all)
        self.transform = transform

    def load_df(self, train, train_all):
        # Load data
        df = pd.read_csv(self.data_dir + "metadata.csv")

        # CV option
        if train_all == False:
            if train == True:
                df = df[df.validation == False].reset_index(drop=True)
            else:
                df = df[df.validation == True].reset_index(drop=True)
        # Train on all data option
        else:
            if train == True:
                df = df.reset_index(drop=True)
            else:
                return None, None

        # Create img paths and one hot labels
        imgs = df.file_name

        # Mapping
        class_map = {'vertical_bar': 0, 'horizontal_bar': 1, 'line': 2, 'scatter': 3, 'dot': 4}
        labels = F.one_hot(
            torch.from_numpy(df.chart_type.map(class_map).values), 
            num_classes=5,
            ).type(torch.DoubleTensor)
        return imgs, labels

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image = Image.open(self.data_dir + self.imgs.iloc[idx])
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

class BenetechClassifierDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        cache_dir: str,
        model_path: str,
        train_all: bool,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_transform, self.val_transform = self._init_transforms()
    
    def _init_transforms(self):
        # Get dimensions of model
        effnet_mappings = {"B0": 224, "B1": 240, "B2": 260, "B3": 300, "B4": 380}
        if self.hparams.model_path not in effnet_mappings.keys():
            raise ValueError(f"{self.hparams.model_path} is not a valid model")
        dimensions = effnet_mappings[self.hparams.model_path]

        # Set Transforms
        img_transforms = transforms.Compose([
            transforms.Resize(380),
            transforms.CenterCrop(dimensions),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]),
        ])
        return img_transforms, img_transforms
    
    def setup(self, stage):        
        if stage == "fit":
            self.train_dataset = self._dataset(train=True, transform=self.train_transform)
            self.val_dataset = self._dataset(train=False, transform=self.val_transform)
            
    def _dataset(self, train, transform):
        return BenetechClassifierDataset(data_dir=self.hparams.data_dir, train=train, transform=transform, train_all=self.hparams.train_all)
    
    def train_dataloader(self):
        return self._dataloader(self.train_dataset, train=True)
    
    def val_dataloader(self):
        if self.hparams.train_all == False:
            return self._dataloader(self.val_dataset, train=False)
        else:
            return []

    def _dataloader(self, dataset, train=False):
        return torch.utils.data.DataLoader(
            dataset,
            shuffle = train,
            batch_size = self.hparams.batch_size,
            num_workers = self.hparams.num_workers,
            pin_memory = True, # True for when processing is done on CPU
        )

class BenetechClassifierModule(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        cache_dir: str,
        model_save_dir: str,
        model_path: str,
        run_name: str,
        save_model: bool,
        num_classes: int,
        label_smoothing: float,
        epochs: int,
        scheduler: str,
        fast_dev_run: bool,
        lr_min: float,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = self._init_model()
        self.metrics = self._init_metrics()
        self.loss_fn = self._init_loss_fn()

    def _init_model(self):
        if self.hparams.model_path == "B0":
            model = torchvision.models.efficientnet_b0()
        elif self.hparams.model_path == "B1":
            model = torchvision.models.efficientnet_b1()
        elif self.hparams.model_path == "B2":
            model = torchvision.models.efficientnet_b2()
        elif self.hparams.model_path == "B3":
            model = torchvision.models.efficientnet_b3()
        elif self.hparams.model_path == "B4":
            model = torchvision.models.efficientnet_b4()
        else:
            raise ValueError(f"{self.hparams.model_path} is not a valid model")
        
        # Updating model with correct output shape
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, self.hparams.num_classes) 

        # # Get dimensions of model
        # effnet_mappings = {"B0": 224, "B1": 240, "B2": 260, "B3": 300, "B4": 380}
        # dimensions = effnet_mappings[self.hparams.model_path]

        # Print model summary
        # torchinfo.summary(model, input_size=(64, 3, dimensions, dimensions))
        return model
    
    def _init_optimizer(self):
        return optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr,
            )

    def _init_scheduler(self, optimizer):
        if self.hparams.scheduler == "CosineAnnealingLR":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max = self.trainer.estimated_stepping_batches,
                eta_min = self.hparams.lr_min,
                )
        elif self.hparams.scheduler == "CosineAnnealingLRDecay":
            if self.hparams.fast_dev_run == True:
                num_cycles = 1
            else:
                num_cycles = 5
            return CosineLRScheduler(
                optimizer, 
                t_initial = self.trainer.estimated_stepping_batches // num_cycles, 
                cycle_decay = 0.75, 
                cycle_limit = num_cycles, 
                lr_min = self.hparams.lr_min,
                )
        else:
            raise ValueError(f"{self.hparams.scheduler} is not a valid scheduler.")
        
    def lr_scheduler_step(self, scheduler, optimizer_idx):
        scheduler.step(
            epoch=self.global_step
        )
    
    def _init_loss_fn(self):
        return nn.CrossEntropyLoss(
            label_smoothing = self.hparams.label_smoothing,
            )
    
    def _init_metrics(self):
        metrics = {
            "acc": torchmetrics.Accuracy(
                task = "multiclass",
                num_classes = self.hparams.num_classes,
            ),
        }
        
        metric_collection = torchmetrics.MetricCollection(metrics)
        return torch.nn.ModuleDict(
            {
                "train_metrics": metric_collection.clone(prefix="train_"),
                "val_metrics": metric_collection.clone(prefix="val_"),
            }
        )

    def configure_optimizers(self):
        optimizer = self._init_optimizer()
        scheduler = self._init_scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
    
    def forward(self, x):
        return self.model(x)
    
    def _shared_step(self, batch, stage, batch_idx):
        x, y = batch
        y_logits = self(x) # Raw logits
        loss = self.loss_fn(y_logits, y)

        y_preds = torch.argmax(y_logits, dim=1) # Preds
        self.metrics[f"{stage}_metrics"](y_preds, torch.argmax(y, dim=1))
        self._log(stage, loss, batch_size=len(x))
        return loss
    
    def validation_step(self, batch, batch_idx):
        pred = self._shared_step(batch, "val", batch_idx)
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train", batch_idx)
    
    def _log(self, stage, loss, batch_size):
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=batch_size)
        self.log_dict(self.metrics[f"{stage}_metrics"], prog_bar=True, batch_size=batch_size)

    def on_train_end(self):
        if self.hparams.save_model == True:
            torch.save(self.model.state_dict(), "{}{}_c.pt".format(self.hparams.model_save_dir, self.hparams.run_name))
        return