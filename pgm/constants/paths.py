"""
Contains paths to various files and directories all given with respect
to the root directory of the project.
"""
import os
from pathlib import Path

# Base paths for the project.
ROOT_PATH = Path(__file__).parent.parent.parent
DATA_PATH = os.path.join(ROOT_PATH, "data")
SCRIPTS_PATH = os.path.join(ROOT_PATH, "scripts")
ETC_PATH = os.path.join(ROOT_PATH, "etc")


# MNIST DATA PATH
IMDB_PATH = os.path.join(DATA_PATH, "imdb_reviews.csv")
IMDB_VOCAB_PATH = os.path.join(DATA_PATH, "aclImdb", "imdb.vocab")

# SVD IMDB DATA
SVD_IMDB_PATH = os.path.join(ETC_PATH, "svd_imdb_100/imdb_svd_data.pkl")
BOW_IMDB_PATH = os.path.join(ETC_PATH, "svd_imdb_100/imdb_bow_data.pkl")
