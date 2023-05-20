from transformers import AutoProcessor, Pix2StructForConditionalGeneration
import requests
import torch
from PIL import Image

class CFG:
    img_path = "https://www.jmp.com/en_nl/statistics-knowledge-portal/exploratory-data-analysis/bar-chart/_jcr_content/par/styledcontainer_2069/par/image.img.png/1594745266124.png"
    # model_path = "google/deplot"
    model_path = "/data/bartley/gpu_test/models/azure-firefly-17.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_TOKENS = 512
    MAX_PATCHES = 1024
    MAX_LENGTH = 512


def process_prediction(processor, model, fp16=False):
    image = Image.open(requests.get(CFG.img_path, stream=True).raw)
    inputs = processor(images=image, text="", return_tensors="pt").to(CFG.device)

    # **inputs passes in kwargs from dict(flattened_patches, attention_mask)
    predictions = model.generate(
        **inputs,
        max_new_tokens=CFG.MAX_LENGTH,
        early_stopping=True,
        use_cache=True,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
        )
    b = processor.decode(predictions[0], skip_special_tokens=True)
    print(b)
    return b

def predict_img(fp16=False):
    """
    Function to predict the data for a single image.
    """
    # model = Pix2StructForConditionalGeneration.from_pretrained("google/deplot").to(device)
    processor = AutoProcessor.from_pretrained("google/deplot")

    if fp16 == False:
        # 3000Mb
        # ----- Full Precision (FP32) -----
        model = Pix2StructForConditionalGeneration.from_pretrained(
            CFG.model_path, 
            is_vqa=False,
            )
        # Source: https://www.kaggle.com/competitions/benetech-making-graphs-accessible/discussion/406250
        model.config.text_config.is_decoder=True
        model = model.to(CFG.device) # Send to GPU

        b = process_prediction(processor, model)
        with open("./test.txt", "a+") as f:
            f.write(b)


    elif fp16 == True:
        # 2273 MB
        # ----- FP 16 -----
        with torch.autocast("cuda"):
            model = Pix2StructForConditionalGeneration.from_pretrained(
                CFG.model_path, 
                is_vqa=False,
                revision="fp16",
                torch_dtype=torch.float16,
                )
            # Source: https://www.kaggle.com/competitions/benetech-making-graphs-accessible/discussion/406250
            model.config.text_config.is_decoder=True
            model = model.to(CFG.device) # Send to GPU

            b = process_prediction(processor, model, fp16=True)
            with open("./test.txt", "a+") as f:
                f.write(b)

def main():
    predict_img(fp16=False)
    predict_img(fp16=True)

if __name__ == "__main__":
    main()
