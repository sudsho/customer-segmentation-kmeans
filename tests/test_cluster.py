"""sanity checks for the cluster module."""
import numpy as np
from src.cluster import fit_kmeans, elbow


def _toy_blobs():
    np.random.seed(0)
    a = np.random.randn(30, 2) + np.array([0, 0])
    b = np.random.randn(30, 2) + np.array([5, 5])
    c = np.random.randn(30, 2) + np.array([-5, 5])
    return np.vstack([a, b, c])


def test_fit_kmeans_returns_correct_k():
    X = _toy_blobs()
    km = fit_kmeans(X, k=3, random_state=0)
    assert len(set(km.labels_)) == 3
    assert km.cluster_centers_.shape == (3, 2)


def test_elbow_inertia_decreases():
    X = _toy_blobs()
    sweep = elbow(X, 2, 5, random_state=0)
    inertias = [v for _, v in sweep]
    # inertia must be monotonically non-increasing as k grows
    for i in range(1, len(inertias)):
        assert inertias[i] <= inertias[i - 1] + 1e-6
