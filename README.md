# customer-segmentation-kmeans

Mall customer segmentation using K-Means clustering. Trains on the Kaggle
Mall Customers dataset, picks k via elbow + silhouette, profiles each
segment, and serves a Plotly Dash dashboard.

[![Build Status](https://travis-ci.org/sudsho/customer-segmentation-kmeans.svg?branch=main)](https://travis-ci.org/sudsho/customer-segmentation-kmeans)

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

(screenshot placeholder; png lives at `docs/dashboard.png` in the deploy.)

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

CI runs on Travis (`.travis.yml`).

## license

MIT.
