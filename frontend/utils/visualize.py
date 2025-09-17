from typing import Dict, Any
import plotly.graph_objects as go


def probability_bar_chart(results: Dict[str, Dict[str, Any]]) -> go.Figure:
    diseases = [k.replace("_", " ").title() for k in results.keys()]
    probs = [v.get("probability", 0.0) for v in results.values()]
    thresholds = [v.get("threshold", 0.5) for v in results.values()]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=diseases,
            y=probs,
            name="Probability",
            marker_color="#1f77b4",
            hovertemplate="%{x}: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=diseases,
            y=thresholds,
            name="Threshold",
            mode="lines+markers",
            line=dict(color="#d62728", dash="dash"),
            hovertemplate="%{x} threshold: %{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        yaxis=dict(title="Probability", range=[0, 1]),
        xaxis=dict(title="Disease"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=20, b=20),
        height=400,
    )
    return fig
