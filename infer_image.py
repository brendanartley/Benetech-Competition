from transformers import AutoProcessor, Pix2StructForConditionalGeneration
import requests
import torch
from PIL import Image

"""
Sample script to use deplot for inference.
"""

class CFG:
    img_path = "https://www.tutorialspoint.com/matplotlib/images/matplotlib_bar_plot.jpg"
    img_path = "https://www.w3schools.com/python/img_matplotlib_bars1.png"
    model_path = "google/deplot"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_TOKENS = 512
    MAX_PATCHES = 1024
    MAX_LENGTH = 512

def process_prediction(processor, model, fp16=False):
    image = Image.open(requests.get(CFG.img_path, stream=True).raw)
    # image = Image.open(CFG.img_path)
    print("image shape: ", image.size)
    inputs = processor(images=image, text="", return_tensors="pt", max_patches=CFG.MAX_PATCHES).to(CFG.device)

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
    return b

def predict_img(fp16=False):
    """
    Function to predict the data for a single image.
    """
    # model = Pix2StructForConditionalGeneration.from_pretrained("google/deplot").to(device)
    processor = AutoProcessor.from_pretrained(CFG.model_path)

    if fp16 == False:
        # 3000Mb
        # ----- Full Precision (FP32) -----
        model = Pix2StructForConditionalGeneration.from_pretrained(
            CFG.model_path, 
            is_vqa=False,
            )
        model = model.to(CFG.device) # Send to GPU

        b = process_prediction(processor, model)
        print(b)
        with open("./test.txt", "w+") as f:
            f.write(b)


    elif fp16 == True:
        # 2273 MB
        # ----- FP 16 -----
        with torch.autocast("cuda"):
            model = Pix2StructForConditionalGeneration.from_pretrained(
                CFG.model_path, 
                is_vqa=False,
                torch_dtype=torch.float16,
                )
            model = model.to(CFG.device) # Send to GPU

            b = process_prediction(processor, model, fp16=True)
            with open("./test.txt", "w+") as f:
                f.write(b)

def main():
    predict_img(fp16=False)

if __name__ == "__main__":
    main()
