"""
Small utilities shared by all the files: where to put things, how to download them.
"""

import os
import urllib.request
import torch

def get_base_dir():
    # everything we download or train lives here, override with an env var if you like
    base_dir = os.environ.get("NANOCHAT_BASE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "nanochat-posttrain"))
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

def download_file(url, filename):
    # download url into the base dir (once), return the local path
    path = os.path.join(get_base_dir(), filename)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, path + ".tmp")
        os.rename(path + ".tmp", path) # rename at the end so a crash never leaves half a file
    return path

def autodetect_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
