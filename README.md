# customer-segmentation-kmeans

Mall customer segmentation using K-Means clustering. Trains on the Kaggle
Mall Customers dataset, picks k via elbow + silhouette, profiles each
segment, and serves a Plotly Dash dashboard.

[![Build Status](https://travis-ci.org/sudsho/customer-segmentation-kmeans.svg?branch=main)](https://travis-ci.org/sudsho/customer-segmentation-kmeans)

## Quick start (runs offline)

No network needed. The dataset is bundled (`data/Mall_Customers.csv`), and the
smoke synthesizes an equivalent frame if the CSV is ever missing. The smoke
builds features, fits KMeans (printing inertia, silhouette and cluster sizes),
profiles each segment, then exercises the predict path on a sample customer.

```
python scripts/smoke.py
```

Real output:

```
=== customer-segmentation-kmeans smoke ===
[data] loading bundled dataset: .../data/Mall_Customers.csv
[data] 200 customers, source=bundled
[features] ['Age', 'Annual Income (k$)', 'Spending Score (1-100)'] -> scaled matrix (200, 3)
[train] fit KMeans k=5
[train] inertia    = 168.25
[train] silhouette = 0.4166
[train] cluster sizes:
          cluster 0: 20 customers
          cluster 1: 54 customers
          cluster 2: 40 customers
          cluster 3: 39 customers
          cluster 4: 47 customers
[profile] per-cluster personas:
          cluster 0: thrifty              (n=20)
          cluster 1: average              (n=54)
          cluster 2: premium              (n=40)
          cluster 3: careful high-income  (n=39)
          cluster 4: average              (n=47)
[save] wrote model bundle -> .../artifacts/smoke_kmeans.joblib
[predict] sample customer {'age': 30, 'income': 85, 'spending': 80} -> cluster 2
=== SMOKE PASSED ===
```

With `make` available you can run the same thing as `make smoke`, the unit
tests as `make test`, and the full elbow/silhouette sweep as `make train`.

## problem

Given mall customer data (age, annual income, spending score), find natural
groups so the marketing team can target offers per segment instead of one
generic blast.

## dataset

`data/Mall_Customers.csv` (the classic Kaggle mall customers set, 200 rows).
Columns: `CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100)`.

If you have transactional data instead, `src/rfm.py` has a small RFM
builder (recency, frequency, monetary) that plugs into the same pipeline.

## approach

1. standard scaling on the three numeric features
2. elbow method (k = 2..10) on inertia
3. silhouette score as a sanity check
4. final k chosen from elbow knee + best silhouette
5. cluster profiling (mean age, income, spending; persona label per segment)

## results

5 clusters give the cleanest split on the income vs spending plane:

| cluster | persona               | avg income (k$) | avg spending |
|--------:|-----------------------|----------------:|-------------:|
| 0       | average               | ~55             | ~50          |
| 1       | premium               | ~85             | ~80          |
| 2       | careful high-income   | ~85             | ~20          |
| 3       | young splurger        | ~25             | ~75          |
| 4       | thrifty               | ~25             | ~20          |

(rounded means after a fresh run; exact numbers vary slightly with seed.)

## dashboard

A Plotly Dash app under `dashboard/` shows:

- scatter of income vs spending colored by cluster
- elbow plot
- silhouette score by k
- per-cluster summary table with persona labels

![dashboard screenshot](docs/dashboard.png)

See `docs/dashboard.png` for a screenshot of the running dashboard.

## setup

```
pip install -r requirements.txt
python -m src.cluster --config configs/default.yaml
python dashboard/app.py
```

Open `http://127.0.0.1:8050`.

## deploy

Heroku:

```
heroku create
git push heroku main
```

`Procfile` and `runtime.txt` are committed. `Dockerfile` is also provided
for a portable image.

## tests

```
pytest -q
```

Real output:

```
.....                                                                    [100%]
5 passed in 2.92s
```

CI runs on Travis (`.travis.yml`).

## license

MIT.
