import io
import base64
from typing import Union

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from loguru import logger


# ── Colour mapping ────────────────────────────────────────────
SENTIMENT_COLORS = {
    "positive": "#1D9E75",
    "neutral":  "#EF9F27",
    "negative": "#D85A30",
}


def plot_aspect_radar(
    aspect_scores: dict,
    title: str = "Aspect Sentiment Analysis",
    max_aspects: int = 8,
) -> go.Figure:
    
    if not aspect_scores:
        logger.warning("No aspects to plot")
        return go.Figure()

    sorted_aspects = sorted(
        aspect_scores.items(),
        key=lambda x: abs(x[1].get("score", x[1].get("mean_score", 0))),
        reverse=True
    )[:max_aspects]

    aspects = [a for a, _ in sorted_aspects]
    scores  = [d.get("score", d.get("mean_score", 0)) for _, d in sorted_aspects]

    radar_vals = [(s + 1) / 2 for s in scores]
    radar_vals_closed = radar_vals + [radar_vals[0]]  
    aspects_closed    = aspects + [aspects[0]]

    avg_score = np.mean(scores)
    fill_color = (
        "rgba(29, 158, 117, 0.3)"  if avg_score > 0.1
        else "rgba(216, 90, 48, 0.3)"  if avg_score < -0.1
        else "rgba(239, 159, 39, 0.3)"
    )
    line_color = (
        "#1D9E75" if avg_score > 0.1
        else "#D85A30" if avg_score < -0.1
        else "#EF9F27"
    )

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=radar_vals_closed,
        theta=aspects_closed,
        fill="toself",
        fillcolor=fill_color,
        line=dict(color=line_color, width=2),
        marker=dict(size=6),
        hovertemplate="<b>%{theta}</b><br>Score: %{customdata:.2f}<extra></extra>",
        customdata=scores + [scores[0]],
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=15)),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                ticktext=["Very Neg", "Negative", "Neutral", "Positive", "Very Pos"],
                tickfont=dict(size=9),
            )
        ),
        showlegend=False,
        height=420,
        margin=dict(l=60, r=60, t=60, b=40),
    )

    return fig


def plot_aspect_bars(
    aspect_scores: dict,
    title: str = "Aspect Sentiment Breakdown",
    max_aspects: int = 10,
) -> go.Figure:
    
    if not aspect_scores:
        return go.Figure()

    sorted_aspects = sorted(
        aspect_scores.items(),
        key=lambda x: x[1].get("score", x[1].get("mean_score", 0)),
    )[:max_aspects]

    aspects = [a for a, _ in sorted_aspects]
    scores  = [d.get("score", d.get("mean_score", 0)) for _, d in sorted_aspects]
    colors  = [
        SENTIMENT_COLORS["positive"] if s > 0.1
        else SENTIMENT_COLORS["negative"] if s < -0.1
        else SENTIMENT_COLORS["neutral"]
        for s in scores
    ]

    fig = go.Figure(go.Bar(
        x=scores,
        y=aspects,
        orientation="h",
        marker_color=colors,
        hovertemplate="<b>%{y}</b><br>Score: %{x:.3f}<extra></extra>",
    ))

    fig.add_vline(x=0, line_width=1.5, line_color="gray")
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(range=[-1.1, 1.1], title="Sentiment Score",
                   tickvals=[-1, -0.5, 0, 0.5, 1],
                   ticktext=["Very Neg", "Neg", "Neutral", "Pos", "Very Pos"]),
        yaxis=dict(title=""),
        height=max(300, len(aspects) * 35 + 100),
        margin=dict(l=120, r=40, t=50, b=40),
        plot_bgcolor="white",
    )
    return fig


def plot_sentiment_distribution(results: list[dict], title: str = "Sentiment Distribution") -> go.Figure:

    from collections import Counter
    counts = Counter(r.get("label", r.get("sentiment", "unknown")) for r in results)

    labels = ["positive", "neutral", "negative"]
    values = [counts.get(l, 0) for l in labels]
    colors = [SENTIMENT_COLORS[l] for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        marker_colors=colors,
        hole=0.45,
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, x=0.5),
        showlegend=True,
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def fig_to_base64(fig: go.Figure) -> str:
    img_bytes = fig.to_image(format="png", width=700, height=450)
    return base64.b64encode(img_bytes).decode("utf-8")