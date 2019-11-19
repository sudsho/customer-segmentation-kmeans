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


def cluster_label(row, income_col="Annual Income (k$)", spend_col="Spending Score (1-100)"):
    """human-readable persona name based on income+spending mean."""
    inc = row[income_col]
    spend = row[spend_col]
    if inc >= 70 and spend >= 60:
        return "premium"
    if inc >= 70 and spend < 40:
        return "careful high-income"
    if inc < 40 and spend >= 60:
        return "young splurger"
    if inc < 40 and spend < 40:
        return "thrifty"
    return "average"


def add_personas(summary, **kwargs):
    summary = summary.copy()
    summary["persona"] = summary.apply(lambda r: cluster_label(r, **kwargs), axis=1)
    return summary
