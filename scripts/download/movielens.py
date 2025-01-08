import argparse
import tempfile
from zipfile import ZipFile
import os

import requests


MOVIELENS_URLS = {
    "latest": "http://files.grouplens.org/datasets/movielens/ml-latest.zip",
    "latest-small": "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    "ml-1m": "http://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "ml-10m": "http://files.grouplens.org/datasets/movielens/ml-10m.zip",
    "ml-20m": "http://files.grouplens.org/datasets/movielens/ml-20m.zip"
}

def download_movielens(
        package: str = "latest-small",
        verbose: bool = False,
    ):
    """
    Reference: https://www.codingforentrepreneurs.com/blog/download-the-movielens-dataset-with-python
    """
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../recommender/.data/movielens"
    )
    url = MOVIELENS_URLS.get(package)
    if not url:
        raise Exception(f"Movie lens package: {package} was not found.")
    if verbose is True:
        print(f"Downloading from {url}")
    os.makedirs(output_dir, exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size_in_bytes= int(r.headers.get("content-length", 0))
        with tempfile.NamedTemporaryFile(mode="rb+") as temp_f:
            downloaded = 0
            dl_iteration = 0
            chunk_size = 8192
            total_chunks = total_size_in_bytes / chunk_size if total_size_in_bytes else 100
            for chunk in r.iter_content(chunk_size=chunk_size):
                if verbose is True:
                    downloaded += chunk_size
                    dl_iteration += 1
                    percent = (100 * dl_iteration * 1.0/total_chunks)
                    if dl_iteration % 10 == 0 and percent < 100:
                        print(f"Completed {percent:2f}%")
                    elif percent >= 99.9:
                        print(f"Download completed. Now unzipping...")
                temp_f.write(chunk)
            with ZipFile(temp_f, "r") as zipf:
                zipf.extractall(output_dir)
                if verbose is True:
                    print(f"\n\nUnzipped.\n\nFiles downloaded and unziped to:\n\n{output_dir}")


def setup_args():
    parser = argparse.ArgumentParser(description="Download movielens")
    parser.add_argument("--verbose", default=True, action="store_true")
    parser.add_argument("--package", default="ml-1m", type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = setup_args()
    verbose = args.verbose
    package = args.package
    download_movielens(package=package, verbose=verbose)