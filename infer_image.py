from transformers import AutoProcessor, Pix2StructForConditionalGeneration
import requests
import torch
from PIL import Image

# USE gpu for gen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = Pix2StructForConditionalGeneration.from_pretrained("google/deplot").to(device)
processor = AutoProcessor.from_pretrained("google/deplot")

# 3000Mb
# ----- Full Precision (FP32) -----
model = Pix2StructForConditionalGeneration.from_pretrained(
    "/data/bartley/gpu_test/models/azure-firefly-17.pt", 
    is_vqa=False,
    )
model = model.to(device) # Send to GPU

for i in range(10):
    url = "https://www.jmp.com/en_nl/statistics-knowledge-portal/exploratory-data-analysis/bar-chart/_jcr_content/par/styledcontainer_2069/par/image.img.png/1594745266124.png"

    image = Image.open(requests.get(url, stream=True).raw)

    inputs = processor(images=image, text="", return_tensors="pt").to(device)
    predictions = model.generate(**inputs, max_new_tokens=512)
    print(processor.decode(predictions[0], skip_special_tokens=True))

    b = processor.decode(predictions[0], skip_special_tokens=True)
    with open("./test.txt", "w+") as f:
        f.write(b)

# 2273 MB
# ----- FP 16 -----
with torch.autocast("cuda"):
    model = Pix2StructForConditionalGeneration.from_pretrained(
        "/data/bartley/gpu_test/models/azure-firefly-17.pt", 
        is_vqa=False,
        revision="fp16",
        torch_dtype=torch.float16,
        )
    model = model.to(device) # Send to GPU

    for i in range(5):
        url = "https://www.jmp.com/en_nl/statistics-knowledge-portal/exploratory-data-analysis/bar-chart/_jcr_content/par/styledcontainer_2069/par/image.img.png/1594745266124.png"

        image = Image.open(requests.get(url, stream=True).raw)

        inputs = processor(images=image, text="", return_tensors="pt").to(device)
        predictions = model.generate(**inputs, max_new_tokens=512)
        print(processor.decode(predictions[0], skip_special_tokens=True))

        b = processor.decode(predictions[0], skip_special_tokens=True)
        with open("./test.txt", "w+") as f:
            f.write(b)