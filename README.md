# customer-segmentation-kmeans

Mall customer segmentation using K-Means clustering.

## problem

Given mall customer data (age, annual income, spending score), find natural
groups so the marketing team can target offers per segment instead of one
generic blast.

## dataset

`Mall_Customers.csv` from Kaggle (the classic 200-row mall customers set).
Columns: `CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100)`.

## approach

- standard scaling on numeric features
- elbow method (k = 2..10) on inertia
- silhouette score as a sanity check
- final k chosen from the elbow + silhouette together
- cluster profiling: mean age, income, spending per cluster

## results

(filled in after training. expect 5 clusters on the income-vs-spending pair.)

## dashboard

A small Plotly Dash app under `dashboard/` shows:
- scatter of income vs spending colored by cluster
- elbow plot
- per-cluster summary table

## setup

```
pip install -r requirements.txt
python src/cluster.py --config configs/default.yaml
python dashboard/app.py
```
