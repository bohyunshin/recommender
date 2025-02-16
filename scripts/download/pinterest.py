import requests
import os
from zipfile import ZipFile


OUTPUT_PATH = "pinterest.zip"
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../recommender/.data/pinterest"
)
URL = "https://github.com/bohyunshin/recommender-dataset/raw/refs/heads/main/pinterest/pinterest_iccv.zip"


def download():
    """
    Download a Git LFS file directly from its URL
    """
    response = requests.get(URL, stream=True)
    response.raise_for_status()

    with open(OUTPUT_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"File downloaded successfully to: {OUTPUT_PATH}")

    with ZipFile(OUTPUT_PATH, "r") as zipf:
        zipf.extractall(OUTPUT_DIR)

    print(f"File unzipped successfully to: {OUTPUT_DIR}")

    os.remove(OUTPUT_PATH)


if __name__ == "__main__":
    download()
