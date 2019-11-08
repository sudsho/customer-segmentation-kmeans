"""describe each cluster's stats: mean age/income/spending, count, gender split."""
import pandas as pd


def describe_clusters(df, labels):
    df = df.copy()
    df["cluster"] = labels
    return df
