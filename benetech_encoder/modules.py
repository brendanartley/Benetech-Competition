import torch
import lightning.pytorch as pl
import pandas as pd
from PIL import Image

from transformers import AutoProcessor, Pix2StructForConditionalGeneration
from benetech_encoder.metrics import BenetechMetric

class ImageCaptioningDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, val_repeat_n, processor, max_patches, validation_set, train_all, chart_type, axis):
        self.data_dir = data_dir
        self.val_repeat_n = val_repeat_n
        self.processor = processor
        self.max_patches = max_patches
        self.chart_type = chart_type
        self.imgs, self.labels = self.load_df(validation_set, train_all, axis)

    def __len__(self):
        return len(self.imgs)
    
    def load_df(self, validation_set, train_all, axis):
        # Load data
        df = pd.read_csv(self.data_dir + "metadata.csv")

        # Select chart_type
        class_map = {'l': 'line', 'v': 'vertical_bar', 's': 'scatter', 'h': 'horizontal_bar', 'd': 'dot'}
        df = df[df.chart_type == class_map[self.chart_type]]

        # Cross-Validation option
        if train_all == False:
            if validation_set == True:
                df = df[df.validation == True].reset_index(drop=True)
            else:
                df = df[df.validation == False].reset_index(drop=True)

        # Train on all data option
        else:
            if validation_set == False:

                # Repeat validation indices N times (as these are extracted - not generated)
                if self.chart_type == 'v':
                    df = pd.concat([df[(df.validation == False)].sample(5000)] + [df[(df.validation == True) & (df.chart_type == "vertical_bar")] for _ in range(self.val_repeat_n//2)], ignore_index=True)
                elif self.chart_type == 'l':
                    df = pd.concat([df[(df.validation == False)].sample(5500)] + [df[(df.validation == True) & (df.chart_type == "line")] for _ in range(self.val_repeat_n//2)], ignore_index=True)           
                elif self.chart_type == 's':
                    df = pd.concat([df[(df.validation == False)].sample(5000)] + [df[(df.validation == True) & (df.chart_type == "scatter")] for _ in range(self.val_repeat_n)], ignore_index=True)
                elif self.chart_type == 'h':
                    df = pd.concat([df[(df.validation == False)].sample(5000)] + [df[(df.validation == True) & (df.chart_type == "horizontal_bar")] for _ in range(self.val_repeat_n*2)], ignore_index=True)

                # # Repeat validation indices N times (as these are extracted - not generated)
                # if self.chart_type == 'v':
                #     df = pd.concat([df] + [df[(df.validation == True) & (df.chart_type == "vertical_bar")] for _ in range(self.val_repeat_n//2)], ignore_index=True)
                # elif self.chart_type == 'l':
                #     df = pd.concat([df] + [df[(df.validation == True) & (df.chart_type == "line")] for _ in range(self.val_repeat_n//2)], ignore_index=True)           
                # elif self.chart_type == 's':
                #     df = pd.concat([df] + [df[(df.validation == True) & (df.chart_type == "scatter")] for _ in range(self.val_repeat_n)], ignore_index=True)
                # elif self.chart_type == 'h':
                #     df = pd.concat([df] + [df[(df.validation == True) & (df.chart_type == "horizontal_bar")] for _ in range(self.val_repeat_n*2)], ignore_index=True)
            else:
                return [], []
        
        # Extract IMG path and texts
        imgs = df["file_name"].values

        # Select x or y labels
        if axis == "x":
            idx = 0
        elif axis == "y":
            idx = 1
        else:
            raise ValueError(f"{axis} is not a recognized axis")
        labels = df["text"].apply(lambda value: " <0x0A> ".join([y[idx].strip() for y in [x.split("|") for x in value.split("<0x0A>")]]))

        return imgs, labels
    
    def __getitem__(self, idx):
        # Get IMG/text
        image = Image.open(self.data_dir + self.imgs[idx])
        label = self.labels[idx]

        # Process input
        encoding = self.processor(images=image, text="", return_tensors="pt", max_patches=self.max_patches)        
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        encoding["text"] = label
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
        train_all: bool,
        chart_type: str,
        cache_dir: str,
        axis: str,
        val_repeat_n: int,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.processor = AutoProcessor.from_pretrained("google/deplot", is_vqa=False, cache_dir=cache_dir)
        self.train_transform, self.val_transform = self._init_transforms()
    
    def _init_transforms(self):
        train_transform = None
        val_transform = None
        return train_transform, val_transform
    
    def setup(self, stage):        
        if stage == "fit":
            self.train_dataset = self._dataset(validation_set=False)
            self.val_dataset = self._dataset(validation_set=True)

        if stage == "test":
            self.val_dataset = self._dataset(validation_set=True)
            
    def _dataset(self, validation_set):
        return ImageCaptioningDataset(
            data_dir = self.hparams.data_dir,
            val_repeat_n = self.hparams.val_repeat_n,
            processor = self.processor,
            max_patches = self.hparams.max_patches,
            validation_set = validation_set,
            train_all = self.hparams.train_all,
            chart_type = self.hparams.chart_type,
            axis = self.hparams.axis,
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
        lr: float,
        max_length: int,
        cache_dir: str,
        model_path: str,
        model_save_dir: str,
        run_name: str,
        save_model: bool,
        scheduler: str,
        processor: AutoProcessor,
        lr_min: float,
        chart_type: str,
        axis: str,
        fast_dev_run: bool,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = self._init_model()
        self.processor = processor
        self.metrics = self._init_metrics()

    def _init_model(self):
        # if self.hparams.model_path in ["google/deplot", "google/matcha-base", "google/pix2struct-base"]:
        #     model = Pix2StructForConditionalGeneration.from_pretrained(
        #         self.hparams.model_path, 
        #         is_vqa=False,
        #         cache_dir=self.hparams.cache_dir,
        #     )
        #     return model
        # else:
        #     raise ValueError(f"{self.hparams.model_path} is not a valid model")
        model = Pix2StructForConditionalGeneration.from_pretrained(
            self.hparams.model_path, 
            is_vqa=False,
            cache_dir=self.hparams.cache_dir,
        )
        return model
    
    def _init_optimizer(self):
        return torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.hparams.lr,
            )
    
    def _init_scheduler(self, optimizer):
        if self.hparams.scheduler == "CosineAnnealingLR":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max = self.trainer.estimated_stepping_batches,
                eta_min = self.hparams.lr_min,
                )
        else:
            raise ValueError(f"{self.hparams.scheduler} is not a valid scheduler.")
    
    def _init_metrics(self):
        metrics = {
            "benetech": BenetechMetric(chart_type = self.hparams.chart_type, axis = self.hparams.axis),
        }
        return metrics

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
        if self.hparams.fast_dev_run == False and self.hparams.save_model == True:
            self.model.save_pretrained('{}{}_{}_{}.pt'.format(self.hparams.model_save_dir, self.hparams.run_name, self.hparams.chart_type, self.hparams.axis))
        return