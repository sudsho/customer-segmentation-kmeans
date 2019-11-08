"""kmeans clustering with elbow + silhouette."""
import argparse
import os
import yaml
import joblib
from sklearn.cluster import KMeans

from src.preprocess import load, select_features, scale


def fit_kmeans(X, k, random_state=42):
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    km.fit(X)
    return km


def elbow(X, k_min, k_max, random_state=42):
    """sweep k and return list of (k, inertia)."""
    out = []
    for k in range(k_min, k_max + 1):
        km = fit_kmeans(X, k, random_state)
        out.append((k, km.inertia_))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    df = load(cfg["data_path"])
    X = select_features(df, cfg["features"])
    Xs, scaler = scale(X, cfg["scale_method"])

    print("running elbow sweep...")
    sweep = elbow(Xs, cfg["k_min"], cfg["k_max"], cfg["random_state"])
    for k, inertia in sweep:
        print("k=%d  inertia=%.2f" % (k, inertia))


if __name__ == "__main__":
    main()
