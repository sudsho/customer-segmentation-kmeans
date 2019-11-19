"""basic preprocessing: load csv, scale numeric features."""
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load(path):
    df = pd.read_csv(path)
    # drop any all-null rows defensively
    df = df.dropna(how="all")
    return df


def select_features(df, features):
    sub = df[features].copy()
    # fill any remaining numeric NaN with median to keep kmeans happy
    for c in sub.columns:
        if sub[c].isnull().any():
            sub[c] = sub[c].fillna(sub[c].median())
    return sub


def scale(X, method="standard"):
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    elif method == "none":
        return X.values, None
    else:
        raise ValueError("unknown scale_method: %s" % method)
    Xs = scaler.fit_transform(X)
    return Xs, scaler
