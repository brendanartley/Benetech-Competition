from transformers import AutoProcessor, Pix2StructForConditionalGeneration
from datasets import load_dataset

import torch
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm


class CFG():
    def __init__(self):
        self.ROOT_DIR = "/data/bartley/gpu_test/bartley-benetech-resized/"
        self.METADATA_PATH = "/data/bartley/gpu_test/bartley-benetech-resized/metadata.csv"
        self.MODEL_SAVE_DIR = ""
        self.MAX_PATCHES = 1024
        self.MAX_LENGTH = 512
        self.BATCH_SIZE = 3
        self.EPOCHS = 1
        self.LR = 1e-5


class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(images=item["image"], text="", return_tensors="pt", add_special_tokens=True, max_patches=config.MAX_PATCHES)        
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        encoding["text"] = item["text"]
        return encoding

def collator(batch):
    new_batch = {"flattened_patches":[], "attention_mask":[]}
    texts = [item["text"] for item in batch]

    text_inputs = processor(text=texts, padding="longest", truncation=True, return_tensors="pt", add_special_tokens=True, max_length=config.MAX_LENGTH)
    new_batch["labels"] = text_inputs.input_ids

    for item in batch:
        new_batch["flattened_patches"].append(item["flattened_patches"])
        new_batch["attention_mask"].append(item["attention_mask"])

    new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
    new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])
    return new_batch

if __name__ == "__main__":

    # Get Config
    config = CFG()

    # Load Datasets
    # "imagefolder" automatically loads metadata
    dataset = load_dataset("imagefolder", data_dir=config.ROOT_DIR, split="validation")

    # Load Models
    processor = AutoProcessor.from_pretrained("google/deplot", is_vqa=False)
    model = Pix2StructForConditionalGeneration.from_pretrained("/data/bartley/gpu_test/models/fallen-butterfly-20.pt")
    model.config.text_config.is_decoder=True # Source: https://www.kaggle.com/competitions/benetech-making-graphs-accessible/discussion/406250
    
    # Load dataset
    valid_dataset = ImageCaptioningDataset(dataset, processor)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=config.BATCH_SIZE, collate_fn=collator)

    # Validate?
    total_batches = len(valid_dataloader)
    print("Total batches: ", total_batches)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    with open("./preds1b.txt", "w") as f:
        for idx, batch in tqdm(enumerate(valid_dataloader), total = total_batches):
            labels = batch.pop("labels").to(device)
            flattened_patches = batch.pop("flattened_patches").to(device)
            attention_mask = batch.pop("attention_mask").to(device)

            predictions = model.generate(
                flattened_patches=flattened_patches, 
                attention_mask=attention_mask, 
                max_new_tokens=config.MAX_LENGTH,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id,
                no_repeat_ngram_size=10, #TODO: TEST HOW THIS AFFECTS
                )
            
            for pred in processor.batch_decode(predictions, skip_special_tokens=True):
                print("{}".format(pred), file=f)