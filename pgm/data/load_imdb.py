from lib2to3.pgen2.token import NEWLINE
from multiprocessing.resource_sharer import stop
from turtle import st
from webbrowser import get
import numpy as np
from pgm.constants.paths import IMDB_PATH, SCRIPTS_PATH
from pathlib import Path
import subprocess
import os
import pandas as pd
import nltk
from pgm.constants.paths import IMDB_VOCAB_PATH, SVD_IMDB_PATH, BOW_IMDB_PATH
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle as pkl

import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("stopwords")


def print_space_and_space(lines=5, dashed_line=True):

    for i in range(lines):
        print("\n")

    if dashed_line:
        print("-------------------------")


def get_imdb_dataset(sentiment_to_number=False, dataframe=True):

    # Download dataset if it isn't present
    if not Path(IMDB_PATH).is_file():
        print("Downloading MNIST Dataset")
        subprocess.call(["sh", os.path.join(SCRIPTS_PATH, "imdb.sh")])

    # ensure that dataset is downloaded
    assert Path(IMDB_PATH).is_file() == True, "Dataset downloaded incorrectly"

    # extract data from csv file
    # data = torch.from_numpy(np.genfromtxt(IMDB_PATH, delimiter=","))
    data = pd.read_csv(IMDB_PATH, index_col=0, names=("review", "signal"))
    # print(data.shape)

    # Change sentiment to number
    # pos = 1, neg = 0
    if sentiment_to_number:
        print_space_and_space(lines=2)
        print("Setting sentiment + == 1 ; - == 0")
        data["signal"].replace(to_replace="pos", value=1, inplace=True)
        data["signal"].replace(to_replace="neg", value=0, inplace=True)

    if dataframe:
        return data
    else:
        return data["review"], data["signal"]


def get_imdb_features(
    dimension=1000, load=False, representation="tfidf", min_df=100, save=False
):

    if representation == "tfidf":
        if not Path(SVD_IMDB_PATH).is_file() or not load:
            print(
                "Running truncated SVD on the dataset to reduce dimensionality to %d",
                dimension,
            )
            data = get_imdb_dataset(dataframe=True)
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(data["review"].tolist())
            svd = TruncatedSVD(n_components=dimension, n_iter=7, random_state=42)
            svd.fit(X)
            data_reduced = svd.transform(X)
            return data_reduced

        else:
            print("loading saved features...")
            print_space_and_space(lines=2)
            data_reduced = pkl.load(open(SVD_IMDB_PATH, "rb"))
            return data_reduced

    elif representation == "bow":
        if not Path(BOW_IMDB_PATH).is_file() or not load:
            print(
                "Running BoW the dataset and reducing dimensionality with min_df %d",
                min_df,
            )
            data = get_imdb_dataset(dataframe=True)
            vectorizer = CountVectorizer(min_df=100, stop_words="english")
            X = vectorizer.fit_transform(data["review"].tolist())
            data_reduced = X.todense()  # Dim: reviews X vocab
            # svd = TruncatedSVD(n_components=dimension, n_iter=7, random_state=42)
            # svd.fit(X)
            # data_reduced = svd.transform(X)
            return data_reduced

        else:
            print("loading saved features...")
            print_space_and_space(lines=2)
            data_reduced = pkl.load(open(BOW_IMDB_PATH, "rb"))
            return data_reduced


def load_processed_imdb_data(
    dims=100, load=True, representation="bow", min_df=100, save=True
):
    # data = torch.from_numpy(get_imdb_features())
    data = get_imdb_features(
        dimension=dims,
        load=load,
        representation=representation,
        min_df=min_df,
        save=True,
    )
    _, sentiment = get_imdb_dataset(sentiment_to_number=True, dataframe=False)
    # sentiment = torch.from_numpy(sentiment.values.astype(np.float32))
    sentiment = sentiment.to_numpy(dtype=np.float32)

    assert data.shape[0] == len(sentiment)

    return data, sentiment
