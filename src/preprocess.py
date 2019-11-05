"""basic preprocessing: load csv, scale numeric features."""
import pandas as pd


def load(path):
    df = pd.read_csv(path)
    return df
