"""Offline smoke test for the customer-segmentation-kmeans pipeline.

Runs end to end with NO network and NO external downloads:

  1. loads the bundled Mall_Customers.csv (falls back to a synthetic
     customer frame if the file is missing, so the smoke never needs a
     network fetch),
  2. builds + scales the numeric features,
  3. fits KMeans and prints inertia, silhouette score and cluster sizes,
  4. saves the model bundle and exercises the predict path by assigning a
     sample customer to a cluster, asserting a valid cluster id.

Run it with:  python scripts/smoke.py   (or:  make smoke)
"""
import os
import sys

import numpy as np
import pandas as pd

# make the repo root importable when run as a plain script
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.preprocess import select_features, scale
from src.cluster import fit_kmeans
from src.profile import describe_clusters, cluster_summary, add_personas
from src.predict import assign
from src.io_utils import ensure_dir
from sklearn.metrics import silhouette_score
import joblib

DATA_PATH = os.path.join(ROOT, "data", "Mall_Customers.csv")
MODEL_PATH = os.path.join(ROOT, "artifacts", "smoke_kmeans.joblib")
FEATURES = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
K = 5
SEED = 42


def synthetic_customers(n=200, seed=SEED):
    """Deterministic synthetic mall-customer frame, used only if the bundled
    CSV is missing. Keeps the smoke fully offline in every case."""
    rng = np.random.RandomState(seed)
    # five loose blobs in (age, income, spending) space
    centers = [
        (45, 55, 50),   # average
        (32, 85, 80),   # premium
        (40, 85, 20),   # careful high-income
        (25, 25, 75),   # young splurger
        (48, 25, 20),   # thrifty
    ]
    rows = []
    for i in range(n):
        cage, cinc, cspend = centers[i % len(centers)]
        rows.append({
            "CustomerID": i + 1,
            "Gender": "Male" if i % 2 == 0 else "Female",
            "Age": int(np.clip(cage + rng.randn() * 6, 18, 70)),
            "Annual Income (k$)": float(np.clip(cinc + rng.randn() * 8, 10, 140)),
            "Spending Score (1-100)": float(np.clip(cspend + rng.randn() * 8, 1, 100)),
        })
    return pd.DataFrame(rows)


def load_customers():
    if os.path.exists(DATA_PATH):
        print("[data] loading bundled dataset: %s" % DATA_PATH)
        return pd.read_csv(DATA_PATH), "bundled"
    print("[data] bundled CSV not found; using synthetic offline fallback")
    return synthetic_customers(), "synthetic"


def main():
    print("=== customer-segmentation-kmeans smoke ===")

    df, source = load_customers()
    print("[data] %d customers, source=%s" % (len(df), source))

    # 1. build + scale features
    X = select_features(df, FEATURES)
    Xs, scaler = scale(X, "standard")
    print("[features] %s -> scaled matrix %s" % (FEATURES, Xs.shape))

    # 2. fit KMeans
    km = fit_kmeans(Xs, k=K, random_state=SEED)
    sil = silhouette_score(Xs, km.labels_)
    sizes = pd.Series(km.labels_).value_counts().sort_index()
    print("[train] fit KMeans k=%d" % K)
    print("[train] inertia    = %.2f" % km.inertia_)
    print("[train] silhouette = %.4f" % sil)
    print("[train] cluster sizes:")
    for c, n in sizes.items():
        print("          cluster %d: %d customers" % (int(c), int(n)))

    assert len(set(km.labels_)) == K, "expected %d non-empty clusters" % K
    assert 0.0 < sil <= 1.0, "silhouette out of range: %r" % sil

    # profiling (persona per cluster) so the profile module is exercised too
    labelled = describe_clusters(df, km.labels_)
    summary = add_personas(cluster_summary(labelled, FEATURES))
    print("[profile] per-cluster personas:")
    for _, row in summary.iterrows():
        print("          cluster %d: %-20s (n=%d)"
              % (int(row["cluster"]), row["persona"], int(row["count"])))

    # 3. save the model bundle
    ensure_dir(os.path.dirname(MODEL_PATH))
    joblib.dump({"model": km, "scaler": scaler, "features": FEATURES}, MODEL_PATH)
    print("[save] wrote model bundle -> %s" % MODEL_PATH)

    # 4. exercise the predict path on a sample customer
    sample = dict(age=30, income=85, spending=80)  # a premium-looking customer
    cluster_id = assign(MODEL_PATH, sample["age"], sample["income"], sample["spending"])
    print("[predict] sample customer %s -> cluster %d" % (sample, cluster_id))
    assert isinstance(cluster_id, int), "cluster id must be an int"
    assert 0 <= cluster_id < K, "cluster id %d out of range [0,%d)" % (cluster_id, K)

    print("=== SMOKE PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
