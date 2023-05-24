import torch
import pytorch_lightning as pl
import pandas as pd
from PIL import Image

from datasets import load_dataset
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
from benetech_decoder.metrics import BenetechMetric

class ImageCaptioningDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, processor, max_patches):
        self.dataset = dataset
        self.processor = processor
        self.max_patches = max_patches

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(images=item["image"], text="", return_tensors="pt", max_patches=self.max_patches)        
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
        cache_dir: str,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.processor = AutoProcessor.from_pretrained(processor_path, is_vqa=False, cache_dir=cache_dir)
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
                dataset = load_dataset("imagefolder", data_dir=self.hparams.data_dir, split="train", cache_dir=self.hparams.cache_dir),
                processor = self.processor,
                max_patches = self.hparams.max_patches,
            )
        else:
            return ImageCaptioningDataset(
                dataset = load_dataset("imagefolder", data_dir=self.hparams.data_dir, split="validation", cache_dir=self.hparams.cache_dir),
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

        # text_inputs = self.processor(
        #     text = texts, 
        #     return_tensors = "pt", 
        #     add_special_tokens = True,
        #     max_length = self.hparams.max_length,
        #     )

        text_inputs = self.processor(
            text = texts, 
            return_tensors = "pt", 
            padding = "longest",
            truncation = True,
            add_special_tokens = True,
            max_length = self.hparams.max_length,
            )
        
        new_batch["labels"] = text_inputs.input_ids
        new_batch["texts"] = texts

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
        cache_dir: str,
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
        self.validation_step_outputs = []
        self.metrics = self._init_metrics()

    def _init_model(self):
        if self.hparams.model_path in ["google/deplot", "google/matcha-base", "google/pix2struct-base"]:
            model = Pix2StructForConditionalGeneration.from_pretrained(
                self.hparams.model_path, 
                is_vqa=False,
                cache_dir=self.hparams.cache_dir,
                )
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
            "benetech": BenetechMetric(),
        }
        return metrics

    def configure_optimizers(self):
        optimizer = self._init_optimizer()
        return {
            "optimizer": optimizer,
        }
    
    def validation_step(self, batch, batch_idx):
        pred = self._shared_step(batch, "valid", batch_idx)
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train", batch_idx)
    
    def _shared_step(self, batch, stage, batch_idx):
        labels = batch.pop("labels")
        flattened_patches = batch.pop("flattened_patches")
        attention_mask = batch.pop("attention_mask")

        # Make Predictions
        outputs = self.model(
            flattened_patches=flattened_patches,
            attention_mask=attention_mask,
            labels=labels,
            )
        
        # Log Loss
        loss = outputs.loss
        self._log(stage, loss, batch_size=len(labels))

        if stage == "valid":

            # Generate text for Benetech Scoring
            predictions = self.model.generate(
                flattened_patches=flattened_patches, 
                attention_mask=attention_mask, 
                max_new_tokens=self.hparams.max_length,
                early_stopping=True,
                use_cache=True,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                )
            
            # Decode texts and get ground truths
            predictions = self.processor.batch_decode(predictions, skip_special_tokens=True)
            ground_truths = batch.pop("texts")

            # Log predictions on first batch
            if batch_idx == 0:
                for gt, pred in zip(ground_truths, predictions):
                    print("True: ", gt)
                    print("Pred: ", pred)

            # Compute benetech score
            self.metrics["benetech"](ground_truths, predictions)
            self.log("benetech_score", self.metrics["benetech"].compute(), on_epoch=True, batch_size=len(labels))
        else:
            return loss
    
    def _log(self, stage, loss, batch_size):
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=batch_size)

    def on_train_end(self):
        if self.hparams.save_model == True:
            self.model.save_pretrained('{}{}.pt'.format(self.hparams.model_save_dir, self.hparams.run_name))
        return