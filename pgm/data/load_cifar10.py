"""
Functions to load the CIFAR-10 dataset
"""

import numpy as np
from pgm.constants.paths import CIFAR10_PATH, SCRIPTS_PATH
from pathlib import Path
import subprocess
import os
import pandas as pd
import pickle as pkl


def unpickle(file):
    with open(file, "rb") as fo:
        dict = pkl.load(fo, encoding="bytes")
    return dict


def is_cifar10() -> bool:
    if not len(os.listdir(CIFAR10_PATH)) > 0:
        print("no files")
        return False

    batch_files = 0
    for file in os.listdir(CIFAR10_PATH):
        if file.startswith("data_batch"):
            batch_files += 1

    if batch_files != 5:
        return False

    return True


def get_cifar10_dataset(batch_number=1):
    """
    Returns two numpy arrays

    images : a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image.
            The first 1024 entries contain the red channel values, the next 1024 the green, and the
            final 1024 the blue. The image is stored in row-major order, so that the first 32
            entries of the array are the red channel values of the first row of the image.

    labels : a list of 10000 numbers in the range 0-9.
            The number at index i indicates the label of the ith image in the array data.
    """

    # Download dataset if it isn't present
    if not is_cifar10():
        print("Downloading CIFAR-10 Dataset")
        subprocess.call(["sh", os.path.join(SCRIPTS_PATH, "cifar.sh")])

    # ensure that dataset is downloaded
    assert is_cifar10() == True, "Dataset downloaded incorrectly"

    # extract data from pickle file
    print("Extracting data from pickle...")
    data = unpickle(os.path.join(CIFAR10_PATH, "data_batch_" + str(batch_number)))

    # dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
    images = np.asarray(data[b"data"])
    labels = np.asarray(data[b"labels"])
    print("Data loaded")

    print("Dataset Dimensions: ", images.shape)
    print("Labels Dimensions: ", labels.shape)

    return images, labels
