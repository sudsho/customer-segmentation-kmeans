"""kmeans clustering with elbow + silhouette."""
import argparse
import os
import yaml
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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


def silhouette_sweep(X, k_min, k_max, random_state=42):
    out = []
    for k in range(max(2, k_min), k_max + 1):
        km = fit_kmeans(X, k, random_state)
        s = silhouette_score(X, km.labels_)
        out.append((k, s))
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

    print("running silhouette sweep...")
    sil = silhouette_sweep(Xs, cfg["k_min"], cfg["k_max"], cfg["random_state"])
    for k, s in sil:
        print("k=%d  silhouette=%.4f" % (k, s))

    final_k = cfg.get("final_k", 5)
    print("fitting final model with k=%d" % final_k)
    km = fit_kmeans(Xs, final_k, cfg["random_state"])

    out_path = cfg.get("model_path", "artifacts/kmeans.joblib")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump({"model": km, "scaler": scaler, "features": cfg["features"]}, out_path)
    print("saved to %s" % out_path)


if __name__ == "__main__":
    main()
