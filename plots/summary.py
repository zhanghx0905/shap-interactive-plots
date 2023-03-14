import json
from typing import Optional

import numpy as np
import plotly.graph_objs as go
import shap

from .utils import RED_BLUE, NumpyEncoder, get_colors, get_minmax, reorder


def jitter(shaps):
    """Copy from shap.summary_plot"""
    N = len(shaps)
    row_height = 0.4
    nbins = 100
    quant = np.round(
        nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8)
    )
    inds = np.argsort(quant + np.random.randn(N) * 1e-6)
    layer = 0
    last_bin = -1
    ys = np.zeros(N)
    for ind in inds:
        if quant[ind] != last_bin:
            layer = 0
        ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
        layer += 1
        last_bin = quant[ind]
    ys *= 0.9 * (row_height / np.max(ys + 1))
    return ys


def summary_plotly(
    shap_values,
    features,
    feature_names=None,
    max_display=20,
    nsample=500,
    save_to: Optional[str] = None,
    verbose: bool = False,
):
    if len(shap_values.shape) > 2:
        raise ValueError(
            "The beeswarm plot does not support plotting explanations with instances that have more "
            "than one dimension!"
        )
    # fallback feature names
    if feature_names is None:
        feature_names = np.array([f"Feature {i}" for i in range(shap_values.shape[0])])

    feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
    feature_order = feature_order[-min(max_display, len(feature_order)) :]
    feature_names = reorder(feature_names, feature_order)
    sampleidx = np.random.choice(shap_values.shape[0], size=nsample)
    shap_values = shap_values[sampleidx]
    data = []
    for yi, idx in enumerate(feature_order):
        x = shap_values[:, idx]
        y = yi + jitter(x)  # np.random.normal(0, 0.05, x.shape)
        vmin, vmax = get_minmax(features[:, idx])
        f = features[sampleidx, idx]
        data.append(
            go.Scatter(
                x=x,
                y=y,
                text=f.astype(str),
                hovertemplate="shapley value: %{x}<br>feature: %{text}<extra></extra>",
                mode="markers",
                marker=dict(
                    color=get_colors(f, vmin, vmax),
                    showscale=False,
                ),
            )
        )
    fig = go.Figure(data=data)
    fig.update_layout(
        height=800,
        # width=500,
        xaxis_title="SHAP value (impact on model output)",
        hovermode="closest",
        showlegend=False,
        yaxis_tickvals=list(range(len(feature_names))),
        yaxis_ticktext=feature_names,
        margin=dict(t=20),
        xaxis=dict(zeroline=True, showline=True, showgrid=False),
        # yaxis=dict(zeroline=False, showline=False, showgrid=False),
    )
    colorbar_trace = go.Scatter(
        x=[None],
        y=[None],
        marker=dict(
            colorscale=RED_BLUE,
            showscale=True,
            cmin=0,
            cmax=255,
            colorbar=dict(
                thickness=5,
                tickvals=[0, 255],
                ticktext=["Low", "High"],
                outlinewidth=0,
                title=dict(
                    font=dict(size=16),
                    text="Feature value",
                    side="right",
                ),
            ),
        ),
        hoverinfo="none",
    )
    fig.add_trace(colorbar_trace)
    if verbose:
        fig.show()
    if save_to is not None:
        with open(save_to, "w", encoding="utf8") as f:
            json.dump(fig.to_plotly_json(), f, cls=NumpyEncoder)
