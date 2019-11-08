"""describe each cluster's stats: mean age/income/spending, count, gender split."""
import pandas as pd


def describe_clusters(df, labels):
    df = df.copy()
    df["cluster"] = labels
    return df


def cluster_summary(df_with_labels, numeric_cols):
    """return a dataframe with mean of numeric_cols and a count, per cluster."""
    g = df_with_labels.groupby("cluster")
    summary = g[numeric_cols].mean().round(2)
    summary["count"] = g.size()
    return summary.reset_index()
