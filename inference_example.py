from transformers import AutoProcessor, Pix2StructForConditionalGeneration
import requests
from PIL import Image

model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-base")
processor = AutoProcessor.from_pretrained("google/pix2struct-base")

url = "https://www.tutorialspoint.com/matplotlib/images/matplotlib_bar_plot.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, text="Generate underlying data table of the figure below:", return_tensors="pt")
predictions = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(predictions[0], skip_special_tokens=True))

b = processor.decode(predictions[0], skip_special_tokens=True)
with open("./test.txt", "w+") as f:
    f.write(b)