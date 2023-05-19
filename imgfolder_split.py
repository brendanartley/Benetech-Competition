import pandas as pd
import os, shutil
from tqdm import tqdm

class config:
    TOP_DIR = "/data/bartley/gpu_test/bartley-benetech-resized/"

# Load metadata
df = pd.read_csv(config.TOP_DIR + "metadata.csv")
print(df.head())

# Check processing has not already been done
if not os.path.exists(config.TOP_DIR + "512x512"):
    raise ValueError("Already completed data split.")

# Create train/valid DIR
for val in ["train", 'validation']:
    path = config.TOP_DIR + val
    if os.path.exists(path) == False: 
        os.mkdir(path)

# Assign value to specific folder
fpaths = [0]*len(df)
for i, (img_path, validation) in tqdm(enumerate(df[['file_name', 'validation']].values), total=len(df)):
    if validation == 1:
        os.replace(config.TOP_DIR + "512x512/" + img_path, config.TOP_DIR + "validation/" + img_path)
        fpaths[i] = "validation/" + img_path
    else:
        os.replace(config.TOP_DIR + "512x512/" + img_path, config.TOP_DIR + "train/"+ img_path)
        fpaths[i] = "train/" + img_path
        
df["file_name"] = fpaths
df.to_csv(config.TOP_DIR + "metadata.csv", index=False)

# Remove 512x512 directory
shutil.rmtree(config.TOP_DIR + "512x512")

# ------------- Creating Test DIR w/ top 100 imgs (FOR TESTING) --------------
test_dir = config.TOP_DIR[:-1] + "-small/"
test_df = df.groupby('validation').head(50)

if os.path.exists(test_dir):
    shutil.rmtree(test_dir)
os.mkdir(test_dir)
os.mkdir(test_dir + "validation")
os.mkdir(test_dir + "train")

for img_path, v in test_df[["file_name", "validation"]].values:
    if v == 1:
        shutil.copyfile(config.TOP_DIR + img_path, test_dir + img_path)
    else:
        shutil.copyfile(config.TOP_DIR + img_path, test_dir + img_path)
test_df.to_csv(test_dir + "metadata.csv", index=False)
    
