import torch
from torch import nn
import pytorch_lightning as pl

from datasets import load_dataset
from transformers import AutoProcessor, Pix2StructForConditionalGeneration


class ImageCaptioningDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, processor, max_patches):
        self.dataset = dataset
        self.processor = processor
        self.max_patches = max_patches

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(images=item["image"], text="", return_tensors="pt", add_special_tokens=True, max_patches=self.max_patches)        
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        encoding["text"] = item["text"]
        return encoding

class BenetechDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        processor_path: str,
        batch_size: int,
        max_length: int,
        num_workers: int,
        max_patches: int,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.processor = AutoProcessor.from_pretrained(processor_path, is_vqa=False)

        self.train_transform, self.val_transform = self._init_transforms()
    
    def _init_transforms(self):
        train_transform = None
        val_transform = None
        return train_transform, val_transform
    
    def setup(self, stage):        
        if stage == "fit":
            self.train_dataset = self._dataset(train=True)
            self.val_dataset = self._dataset(train=False)

        if stage == "test":
            self.val_dataset = self._dataset(train=False)
            
    def _dataset(self, train):
        if train == True:
            return ImageCaptioningDataset(
                dataset = load_dataset("imagefolder", data_dir=self.hparams.data_dir, split="train"),
                processor = self.processor,
                max_patches = self.hparams.max_patches,
            )
        else:
            return ImageCaptioningDataset(
                dataset = load_dataset("imagefolder", data_dir=self.hparams.data_dir, split="validation"),
                processor = self.processor,
                max_patches = self.hparams.max_patches,
            )
    
    def train_dataloader(self):
        return self._dataloader(self.train_dataset, train=True)
    
    def val_dataloader(self):
        return self._dataloader(self.val_dataset, train=False)
    
    def test_dataloader(self):
        return self._dataloader(self.val_dataset, train=False)

    def _dataloader(self, dataset, train=False):
        return torch.utils.data.DataLoader(
            dataset,
            shuffle = train,
            batch_size = self.hparams.batch_size,
            collate_fn = self.collator,
            num_workers = self.hparams.num_workers,
            pin_memory = True, # True for when processing is done on CPU
        )
    
    def collator(self, batch):
        """
        Collate function to preprocess inputs.

        Source: https://github.com/huggingface/notebooks/blob/2dd248050b5d7fa6c689ea79907efd1845c73dd8/examples/image_captioning_pix2struct.ipynb
        """
        new_batch = {"flattened_patches":[], "attention_mask":[]}
        texts = [item["text"] for item in batch]

        text_inputs = self.processor(
            text = texts, 
            padding = "longest", 
            truncation = True, 
            return_tensors = "pt", 
            add_special_tokens = True,
            max_length = self.hparams.max_length,
            )
        
        new_batch["labels"] = text_inputs.input_ids

        for item in batch:
            new_batch["flattened_patches"].append(item["flattened_patches"])
            new_batch["attention_mask"].append(item["attention_mask"])

        new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
        new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])
        return new_batch
    
class BenetechModule(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float,
        max_length: int,
        model_path: str,
        model_save_dir: str,
        run_name: str,
        save_model: bool,
        processor: AutoProcessor,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = self._init_model()
        self.processor = processor

    def _init_model(self):
        if self.hparams.model_path == "google/deplot":
            model = Pix2StructForConditionalGeneration.from_pretrained(
                "google/deplot", 
                is_vqa=False,
                )
             # Source: https://www.kaggle.com/competitions/benetech-making-graphs-accessible/discussion/406250
            model.config.text_config.is_decoder = True
            return model
        else:
            raise ValueError(f"{self.hparams.model_path} is not a valid model")
    
    def configure_optimizers(self):
        optimizer = self._init_optimizer()
        return {
            "optimizer": optimizer,
        }
    
    def _init_optimizer(self):
        return torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.hparams.learning_rate,
            )
    
    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "valid")
    
    def training_step(self, batch, stage):
        return self._shared_step(batch, "train")
    
    def _shared_step(self, batch, stage):
        labels = batch.pop("labels")#.to(device)
        flattened_patches = batch.pop("flattened_patches")#.to(device)
        attention_mask = batch.pop("attention_mask")#.to(device)

        # Make Predictions
        outputs = self.model(
            flattened_patches=flattened_patches,
            attention_mask=attention_mask,
            labels=labels
            )
        
        # Log Loss
        loss = outputs.loss
        self._log(stage, loss, batch_size=len(labels))
        return loss
    
    def _log(self, stage, loss, batch_size):
        self.log(f"{stage}_loss", loss, batch_size=batch_size)

    def on_train_end(self):
        if self.hparams.save_model == True:
            self.model.save_pretrained('{}{}.pt'.format(self.hparams.model_save_dir, self.hparams.run_name))
        return