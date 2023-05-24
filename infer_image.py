from transformers import AutoProcessor, Pix2StructForConditionalGeneration
import requests
import torch
from PIL import Image

class CFG:
    img_path = "https://www.tutorialspoint.com/matplotlib/images/matplotlib_bar_plot.jpg"
    img_path = "https://www.w3schools.com/python/img_matplotlib_bars1.png"
    
    img_path = "/data/bartley/gpu_test/bartley-benetech-resized/validation/000b92c3b098.jpg"
    img_path = "https://storage.googleapis.com/kagglesdsdata/competitions/43873/5585780/train/images/000b92c3b098.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20230521%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230521T175113Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=9438728456e718660f873e93382ac0d5f990d901c5085700347561a9353ef7e5dbc2ecf893b985e8b3d4f3aba089712b968f37fdf4d9e905c85d481b20a4da6f1ea3f44c9a306b816ca093632571a32fc447527f33cb52962e1f072eb5bbf2b581cc0c467db6216aa1c1638de42169ff02db973d8b6af9d1e9885eb183d62fd34060e80bb0f4d85f90d8b5e5e71e95c60b1755d3a93a6f16ab4df3a677dadd7c018b480561d4497cbc558a25c15b76c8c5fc41a7863e5df44217baad4e47c0a420eb201e31b863e6175a069aeba16a66bf882e6f7f9e8b91adb15085de69a6f403ad2093c711755b06839339fb5e9b9e5555f5ac0e5a12b146469e9ba476559d"

    img_path = "https://storage.googleapis.com/kagglesdsdata/datasets/3255997/5748653/512x512/0000ae6cbdb1.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20230523%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230523T185358Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=228a1279abf1ac695f5a17f7be760f8d20bf1f851d93085cc9c9355632a737783351be999f6ae2ac5340607f199a6aead24b3217eba7279acb046f737cc57cb57cb8596c36a6268b952ed8e79eaad9265e3e2a44c4e760a70b992d05148df554b136257c075b5f6dd3371a9893832c04923906439a1fc52c948ffb809be755fcd4c3363a30f70b13cc8f32011a2798b32a6af1e5b4a5899ab91a2865e0f30b66dbc495ae19fa291c52fa0d0edba7cbeddad140210eab51bcc090ee810365a8ddbbf124229a78e1e98c1c2ca653d005772af2d1a02284973877be8092bae1a26b9b618a77efb79dc2500da5fbfc3e42a26ffb355f4089a3b003fb0887034b6cd9"

    # model_path = "google/pix2struct-base"
    # model_path = "google/matcha-base"
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
    # predict_img(fp16=True)

if __name__ == "__main__":
    main()
