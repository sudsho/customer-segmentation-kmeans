"""kmeans clustering with elbow + silhouette."""
import argparse
import yaml
from sklearn.cluster import KMeans

from src.preprocess import load, select_features, scale


def fit_kmeans(X, k, random_state=42):
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    km.fit(X)
    return km
