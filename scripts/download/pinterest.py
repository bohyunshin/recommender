import os

import gdown
from zipfile import ZipFile

GOOGLE_FILE_ID = "0B0l8Lmmrs5A_REZXanM3dTN4Y28"
URL = f"https://drive.google.com/uc?id={GOOGLE_FILE_ID}"
OUTPUT_FILE_NAME = "pinterest.zip"

OUTPUT_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../recommender/.data/pinterest"
)

def download():
    gdown.download(URL, output=OUTPUT_FILE_NAME, quiet=False)
    with ZipFile(OUTPUT_FILE_NAME, "r") as zipf:
        zipf.extractall(OUTPUT_DIR)
    os.remove(OUTPUT_FILE_NAME)


if __name__ == "__main__":
    download()