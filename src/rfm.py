"""rfm helpers for transactional customer data.

the mall dataset doesn't have transactions but if a transactions csv is
plugged in (customer_id, order_date, amount), this builds the rfm frame
that we then segment with the same kmeans pipeline.
"""
import pandas as pd


def build_rfm(transactions, snapshot_date=None):
    """transactions has columns customer_id, order_date, amount."""
    df = transactions.copy()
    df["order_date"] = pd.to_datetime(df["order_date"])
    if snapshot_date is None:
        snapshot_date = df["order_date"].max() + pd.Timedelta(days=1)
    snapshot_date = pd.to_datetime(snapshot_date)

    g = df.groupby("customer_id")
    rfm = pd.DataFrame({
        "recency": (snapshot_date - g["order_date"].max()).dt.days,
        "frequency": g.size(),
        "monetary": g["amount"].sum(),
    })
    return rfm.reset_index()
