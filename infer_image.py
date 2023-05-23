from transformers import AutoProcessor, Pix2StructForConditionalGeneration
import requests
import torch
from PIL import Image

class CFG:
    # img_path = "https://storage.googleapis.com/kagglesdsdata/competitions/43873/5585780/train/images/0000ae6cbdb1.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20230521%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230521T175113Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=620075b4d7ac9cbe407f70a93d8e1b89c7dc141b89cf1cd833d9b31ae2acf45411d922d45508ca63403183cd5e924f8b29f0ae1f3e7d31131798ff36db27c236f3747518f857f8f13de8bc0d6485231d0c9bbdb7466cfcfe1c07224facdb7f2fbe5bf6ceeb0c491283c7ad5e3ea4e6f2439fbe28d7d0128ff50e5f5c0f75c89162eeae92124b5488d6fd97ab6f37a19b1a3fbbf94cd16f63db46baf0c216534ea1c748560b4d040d982ba4529008ab80bdf4abacc6b59d3d5ea4f5f736df0936c6399a6037c35bc5d696521932eab7712af5fa2749619aa39f14ea51ae319f0bda3a429ad8f86c260dbf94e11edcbb4a5850b6f553551583e38fb8b4c90fc5a3"
    # img_path = "https://storage.googleapis.com/kagglesdsdata/datasets/3255997/5738945/512x512/0000ae6cbdb1.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20230523%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230523T001417Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=85b4d7568e39effea670eb22858d4a83762ede1eca2cca01e8e7e714afff2aa3661bbcba59693f2479ec7e09154e58ea7d7c383b43d14708f49553fe19317da418bbdc9e4d3a2f56baa877fc05826c8f6a33db4448aeff6d5fe16ddc4e323f154b7fdbb9e9c4b422f570cee972c314b62ea370129a0aff385a4dedbb4e6597f52dd3eb11947616f04ccd1aed2c1c0a4f5ef9bcf5b4be2b18df1eab1608af2ad4a01e2cf2eacdac6abade6ca12822bafd0d33112702486027a52ed6d74152e9b6dccfac4e88913bcb3960ad17f0860160f15d89634edc4dad73883c62890d9f66286ce3f53e78dc1179e680775263736c7d5016b9b2f60e5b75177b88aa69ab42"
    # img_path = "https://www.kaggleusercontent.com/kf/130467576/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..dB_QCN2Y8ef0Y7omQ8w_nw.KQ97nDRbI6cGfhRUyPMALUqWGwnv8v1zESjQDT9iTds3Sz4g3NO30e1QxzBb9-fDec5jWj4FQkBzDwgLEOCEYaxKimOaAJeAi-dKk_yR5NTE8fzePWwW6D0lPXuCFdNMxYPPg_Gd2iEkKL0zV4Sev3DONJBqRKWxZJ-7Ibnlh5arjxEg02DtoEmLhu3S-BP_s6oyHoyurpXi70vR2hU2DDdY6E3rutSR42CETMTVyrDXTFBSvp-NP9o7F1TOmPPbfuffbzCOmw8l0ff4bPqr7BPoVNSu_FxzmBcAFOnPkBPgmKP5m9D5EYPSyakTdmfjVjjaAoPgdUTv3ruAH1bfdVt0OaBbq9hrEGaXv3u4SV2rSjt9k9q7waiFCW9BMm53GAnp22b-fhntC-0mPfC7Zvh62lHxaWcSjLnP5P2gMq6re3jEi2gMXXers8QJF5-UX3NWkgMSvXYymWwrxdXEmM0w6SUzB6T3k_XDeJGFN2MdfclYwuNF6PFSGA53QdWwhnlYTUHvOiaRj8yYQTRWrJqCw4D79pxgx8vWC1wRnW5tnnUGmkmQi8WHPuxovbCj9XTN-EfkTl-VAglgmPkTlKNZwvX8i8kjPTS5nxnARogIZ27I0jPG17KMi37milxb1neOILj9X-A5Tk_tsQFKUeEU9lr3q4jBr58G-BNPMo8.Pplv_YyGFbjrok-axhASyQ/512x512/0003a50817cf.jpg"
    img_path = "https://storage.googleapis.com/kagglesdsdata/competitions/43873/5585780/train/images/00261ed70def.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20230523%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230523T005139Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=7d61c6bcc502822550928056f578fd5b48606c770a39d3d198253116254156cb2fa261d174ba31fa9761eb976949c9a36daa7b7519fddc6c125b25968081508b5dda8e6f78969fce23bff7cd1ef55717906d17e7b34d0d21eb7b189e3c7ecd7f47ad334402102b5b797714ada29fe22db9a03e5c03333e4b402d0e2863ca1fd80ad64467476d31a4e11f1fc9e92a73205cfc4f8481e8b6a975ff86986db996199cd61d67776833fbb28e87adbb4894ecab6d28bef1903b281de7932ce5ea965fe2aec13871cea680c6140cbcd7251c16fa237222c8985c91c3a19980cf476a38431b665fec6300c8c8e99430d9633636fd79f9c88de1674fc394905cadda6b3d"
    
    model_path = "google/deplot"
    # model_path = "/data/bartley/gpu_test/models/sparkling-monkey-42.pt"

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
    processor = AutoProcessor.from_pretrained("google/matcha-base")

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
    # predict_img(fp16=True)

if __name__ == "__main__":
    main()
