"""basic preprocessing: load csv, scale numeric features."""
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load(path):
    df = pd.read_csv(path)
    return df


def select_features(df, features):
    return df[features].copy()


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
