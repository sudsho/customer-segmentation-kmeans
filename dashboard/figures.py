"""build the figures the dashboard renders."""
import plotly.graph_objs as go


def elbow_figure(sweep):
    """sweep is a list of (k, inertia)."""
    ks = [k for k, _ in sweep]
    ys = [v for _, v in sweep]
    return go.Figure(
        data=[go.Scatter(x=ks, y=ys, mode="lines+markers")],
        layout=go.Layout(
            title="elbow plot",
            xaxis={"title": "k"},
            yaxis={"title": "inertia"},
        ),
    )


def silhouette_figure(sil):
    ks = [k for k, _ in sil]
    ys = [v for _, v in sil]
    return go.Figure(
        data=[go.Bar(x=ks, y=ys)],
        layout=go.Layout(
            title="silhouette by k",
            xaxis={"title": "k"},
            yaxis={"title": "silhouette score"},
        ),
    )
