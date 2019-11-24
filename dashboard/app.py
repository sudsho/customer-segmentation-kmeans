"""plotly dash app showing the segmentation results."""
import os
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
import joblib

from src.preprocess import select_features, scale
from src.cluster import elbow, silhouette_sweep
from src.profile import cluster_summary, add_personas
from dashboard.figures import elbow_figure, silhouette_figure


DATA_PATH = os.environ.get("DATA_PATH", "data/Mall_Customers.csv")
MODEL_PATH = os.environ.get("MODEL_PATH", "artifacts/kmeans.joblib")
FEATURES = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]


def load_data():
    df = pd.read_csv(DATA_PATH)
    if os.path.exists(MODEL_PATH):
        bundle = joblib.load(MODEL_PATH)
        scaler = bundle["scaler"]
        feats = bundle["features"]
        X = df[feats].values
        if scaler is not None:
            X = scaler.transform(X)
        df["cluster"] = bundle["model"].predict(X)
    else:
        # fallback so the page still renders before training
        df["cluster"] = 0
    return df


def summary_table(df):
    summary = cluster_summary(df, FEATURES)
    summary = add_personas(summary)
    header = [html.Th(c) for c in summary.columns]
    rows = []
    for _, row in summary.iterrows():
        rows.append(html.Tr([html.Td(row[c]) for c in summary.columns]))
    return html.Table([html.Thead(html.Tr(header)), html.Tbody(rows)])


df = load_data()
df["cluster_str"] = df["cluster"].astype(str)
scatter = px.scatter(
    df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    color="cluster_str",
    hover_data=["Age", "Gender"],
    title="customers by cluster",
)

X_raw = select_features(df, FEATURES)
Xs, _ = scale(X_raw, "standard")
elbow_data = elbow(Xs, 2, 10)
sil_data = silhouette_sweep(Xs, 2, 10)


app = dash.Dash(__name__)
app.title = "mall customer segments"
server = app.server  # for gunicorn / heroku

app.layout = html.Div([
    html.H1("mall customer segments"),
    dcc.Graph(id="scatter", figure=scatter),
    html.Div([
        dcc.Graph(id="elbow", figure=elbow_figure(elbow_data)),
        dcc.Graph(id="silhouette", figure=silhouette_figure(sil_data)),
    ]),
    html.H3("per-cluster averages"),
    summary_table(df),
])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host="0.0.0.0", port=port, debug=False)
