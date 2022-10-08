from tqdm import tqdm
from pgm.constants.paths import DATA_PATH
import os
import pandas as pd

ds = []
for directory, _, files in os.walk(os.path.join(DATA_PATH, "aclImdb")):
    if directory not in (
        os.path.join(DATA_PATH, "aclImdb/test/pos"),
        os.path.join(DATA_PATH, "aclImdb/test/neg"),
        os.path.join(DATA_PATH, "aclImdb/train/pos"),
        os.path.join(DATA_PATH, "aclImdb/train/neg"),
    ):
        continue

    sent = "pos" if directory.endswith("/pos") else "neg"

    for fn in tqdm(files):
        with open(os.path.join(directory, fn), "rt") as f:
            ds.append((f.read(), sent))

df = pd.DataFrame(ds)
print(df.shape)
print(df.head(n=2))
# df.to_csv(os.path.join(DATA_PATH, "imdb_reviews.csv"))
