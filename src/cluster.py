"""kmeans clustering with elbow + silhouette."""
import argparse
import yaml
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
