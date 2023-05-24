import pytorch_lightning as pl
import torchvision
from torchvision import transforms
import torch
import torchmetrics
import torchinfo

import pandas as pd
from PIL import Image

class BenetechClassifierDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        self.data_dir = data_dir
        self.df = self.load_df(train)
        self.transform = transform

    def load_df(self, train):
        df = pd.read_csv(self.data_dir + "metadata.csv")
        if train == True:
            df = df[df.validation == False].reset_index(drop=True)
        else:
            df = df[df.validation == True].reset_index(drop=True)
        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        image = Image.open(self.data_dir + item.file_name)

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "chart_type": item.chart_type,
        }

class BenetechClassifierDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        cache_dir: str,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_transform, self.val_transform = self._init_transforms()
    
    def _init_transforms(self):
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]),
        ])
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]),
        ])
        return train_transform, val_transform
    
    def setup(self, stage):        
        if stage == "fit":
            self.train_dataset = self._dataset(train=True, transform=self.train_transform)
            self.val_dataset = self._dataset(train=False, transform=self.val_transform)
            
    def _dataset(self, train, transform):
        return BenetechClassifierDataset(data_dir=self.hparams.data_dir, transform=transform)
    
    def train_dataloader(self):
        return self._dataloader(self.train_dataset, train=True)
    
    def val_dataloader(self):
        return self._dataloader(self.val_dataset, train=False)

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
        learning_rate: float,
        cache_dir: str,
        model_save_dir: str,
        model_path: str,
        run_name: str,
        save_model: bool,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = self._init_model()
        self.metrics = self._init_metrics()

    def _init_model(self):
        if self.hparams.model_path in ["B0"]:
            model = torchvision.models.efficientnet_b0()
            return model
        else:
            raise ValueError(f"{self.hparams.model_path} is not a valid model")
    
    def _init_optimizer(self):
        return torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.hparams.learning_rate,
            )
    
    def _init_metrics(self):
        metrics = {
            "acc": torchmetrics.classification.MulticlassAccuracy(
                num_classes=5
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
        return {
            "optimizer": optimizer,
        }
    
    def forward(self, x):
        return self.model(x)
    
    def _forward_pass(self, batch):
        #TODO: Sort out what happens from here
        print(batch)
        x, y = batch
        y = y.view(-1)
        y_pred = self(x)
        return x, y, y_pred
    
    def _shared_step(self, batch, stage, batch_idx):
        x, y, y_pred = self._forward_pass(batch)
        loss = self.loss_fn(y_pred, y)
        self.metrics[f"{stage}_metrics"](y_pred, y)
        self._log(stage, loss, batch_size=len(x))
        return loss
    
    def validation_step(self, batch, batch_idx):
        pred = self._shared_step(batch, "valid", batch_idx)
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train", batch_idx)
    
    def _log(self, stage, loss, batch_size):
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=batch_size)

    def on_train_end(self):
        # if self.hparams.save_model == True:
        #     self.model.save_pretrained('{}{}.pt'.format(self.hparams.model_save_dir, self.hparams.run_name))
        return