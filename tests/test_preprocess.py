"""tests for preprocess + profile helpers."""
import pandas as pd
import numpy as np
from src.preprocess import scale, select_features
from src.profile import describe_clusters, cluster_summary


def test_select_features_subsets():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    out = select_features(df, ["a", "c"])
    assert list(out.columns) == ["a", "c"]
    assert len(out) == 3


def test_scale_standard_zero_mean():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0]})
    Xs, scaler = scale(X, "standard")
    np.testing.assert_allclose(Xs.mean(axis=0), [0.0, 0.0], atol=1e-8)


def test_cluster_summary_count_matches():
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [10, 20, 30, 40]})
    labels = [0, 0, 1, 1]
    df_l = describe_clusters(df, labels)
    summary = cluster_summary(df_l, ["x", "y"])
    assert summary["count"].sum() == len(df)
