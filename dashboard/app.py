"""plotly dash app showing the segmentation results."""
import dash
import dash_core_components as dcc
import dash_html_components as html


app = dash.Dash(__name__)
app.title = "mall customer segments"

app.layout = html.Div([
    html.H1("mall customer segments"),
    html.P("dashboard wip"),
])


if __name__ == "__main__":
    app.run_server(debug=True)
