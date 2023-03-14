import json

import numpy as np
import plotly.graph_objects as go
import shap

from .utils import *


def render_waterfall(values, feature_names, E_fx, fx, show=False):
    fig = go.Figure(
        go.Waterfall(
            orientation="h",
            base=E_fx,
            measure=["relative" for _ in range(len(values))],
            y=feature_names,
            x=values,
            connector={
                "mode": "between",
                "line": {"width": 4, "color": "rgb(0, 0, 0)", "dash": "solid"},
            },
        )
    )
    fig.add_vline(
        x=E_fx,
        line_dash="dash",
        annotation_text=f"E[f(x)] = {E_fx:.2f}",
        annotation_position="bottom",
    )
    fig.add_vline(
        x=fx,
        line_dash="dash",
        annotation_text=f"f(x) = {fx:.2f}",
        annotation_position="top",
    )
    if show:
        fig.show()
    return fig.to_plotly_json()


def waterfall_plotly(
    shap_value,
    features,
    E_fx,
    feature_names=None,
    max_display=10,
    save_to=None,
    verbose: bool = False,
):
    fx = E_fx + shap_value.sum()

    # make sure we only have a single explanation to plot
    if len(shap_value.shape) == 2:
        raise Exception(
            "The waterfall_plot can currently only plot a single explanation but a matrix of explanations was passed!"
        )

    # fallback feature names
    if feature_names is None:
        feature_names = np.array([f"Feature {i}" for i in range(len(shap_value))])
    num_features = min(max_display - 1, len(shap_value))
    aggregation = len(shap_value) - num_features  # Whether we need aggregation

    order = np.argsort(-np.abs(shap_value))[:num_features]
    values = reorder(shap_value, order)
    features = reorder(features, order)
    feature_names = reorder(feature_names, order)
    feature_names = [
        f"{fvalue:.2f} {fname}" for (fname, fvalue) in zip(feature_names, features)
    ]
    if aggregation > 0:
        feature_names.append(f"{aggregation} other features")
        values.append(fx - E_fx - np.sum(values))
    feature_names = feature_names[::-1]
    values = values[::-1]

    fig_json = render_waterfall(values, feature_names, E_fx, fx, show=verbose)
    if save_to is not None:
        with open(save_to, "w", encoding="utf8") as f:
            json.dump(fig_json, f)
