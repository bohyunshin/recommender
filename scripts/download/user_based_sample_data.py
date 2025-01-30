import os

import requests

TEXT_FILE_DIRECTORY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../recommender/.data/user_based_sample",
)
TEXT_FILE_NAME = "sample.txt"


def get_raw_file(url) -> str:
    res = requests.get(url)
    if res.status_code == 200:
        return res.text
    else:
        raise ValueError(f"Failed to download text file from {url}")


def write_text_file(text: str, directory: str, file_name: str) -> None:
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, file_name), "w") as f:
        f.write(text)


if __name__ == "__main__":
    # download sample test case from kakao recoteam programming assignments
    URL = "https://raw.githubusercontent.com/kakao/recoteam/refs/heads/master/programming_assignments/mini_reco/testcase/input/input006.txt"
    text = get_raw_file(URL)
    write_text_file(
        text=text,
        directory=TEXT_FILE_DIRECTORY,
        file_name=TEXT_FILE_NAME,
    )
