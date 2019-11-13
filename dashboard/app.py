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
        df["cluster"] = bundle["model"].predict(X).astype(str)
    else:
        df["cluster"] = "0"
    return df


df = load_data()
scatter = px.scatter(
    df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    color="cluster",
    hover_data=["Age", "Gender"],
    title="customers by cluster",
)

# elbow + silhouette computed live for the dashboard
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
])


if __name__ == "__main__":
    app.run_server(debug=True)
