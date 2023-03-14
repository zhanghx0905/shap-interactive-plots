import json
from typing import Union

import numpy as np
import plotly.graph_objects as go
import shap

from .utils import RED_BLUE, NumpyEncoder, get_colors


def decision_data(
    base_value: Union[np.ndarray, float],
    shap_values: np.ndarray,
    features=None,
    feature_names=None,
    max_display=20,
):
    # code taken from force_plot. auto unwrap the base_value
    if type(base_value) == np.ndarray and len(base_value.shape) == 1:
        base_value = base_value[0]

    if isinstance(base_value, list) or isinstance(shap_values, list):
        raise TypeError(
            "Looks like multi output. Try base_value[i] and shap_values[i], "
            "or use shap.multioutput_decision_plot()."
        )

    # validate shap_values
    if not isinstance(shap_values, np.ndarray):
        raise TypeError(
            "The shap_values arg is the wrong type. Try explainer.shap_values()."
        )

    # calculate the various dimensions involved (observations, features, interactions, display, etc.
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)
    observation_count = shap_values.shape[0]
    feature_count = shap_values.shape[1]

    # code taken from force_plot. convert features from other types.
    if str(type(features)) == "<class 'pandas.core.frame.DataFrame'>":
        if feature_names is None:
            feature_names = features.columns.to_list()
        features = features.values
    elif str(type(features)) == "<class 'pandas.core.series.Series'>":
        if feature_names is None:
            feature_names = features.index.to_list()
        features = features.values
    elif isinstance(features, list):
        if feature_names is None:
            feature_names = features
        features = None
    elif features is not None and features.ndim == 1 and feature_names is None:
        feature_names = features.tolist()
        features = None

    # the above code converts features to either None or np.ndarray. if features is something else at this point,
    # there's a problem.
    if not isinstance(features, (np.ndarray, type(None))):
        raise TypeError("The features arg uses an unsupported type.")
    if (features is not None) and (features.ndim == 1):
        features = features.reshape(1, -1)

    # validate/generate feature_names. at this point, feature_names does not include interactions.
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(feature_count)]
    elif len(feature_names) != feature_count:
        raise ValueError(
            "The feature_names arg must include all features represented in shap_values."
        )
    elif not isinstance(feature_names, (list, np.ndarray)):
        raise TypeError("The feature_names arg requires a list or numpy array.")

    feature_idx = np.argsort(np.sum(np.abs(shap_values), axis=0))
    # show last 20 features in descending order.
    feature_display_range = slice(-1, -1 - max_display, -1)

    a = feature_display_range.indices(feature_count)

    a = (a[1] + 1, a[0] + 1, 1)
    feature_display_count = a[1] - a[0]
    shap_values = shap_values[:, feature_idx]
    if a[0] == 0:
        cumsum = np.ndarray(
            (observation_count, feature_display_count + 1), shap_values.dtype
        )
        cumsum[:, 0] = base_value
        cumsum[:, 1:] = base_value + np.nancumsum(shap_values[:, 0 : a[1]], axis=1)
    else:
        cumsum = base_value + np.nancumsum(shap_values, axis=1)[:, (a[0] - 1) : a[1]]

    # Select and sort feature names and features according to the range selected above
    feature_names = np.array(feature_names)
    feature_names_display = feature_names[feature_idx[a[0] : a[1]]].tolist()
    feature_names = feature_names[feature_idx].tolist()
    features_display = (
        None if features is None else features[:, feature_idx[a[0] : a[1]]]
    )

    xmin = np.min((cumsum.min(), base_value))
    xmax = np.max((cumsum.max(), base_value))
    # create a symmetric axis around base_value
    a, b = (base_value - xmin), (xmax - base_value)
    if a > b:
        xlim = (base_value - a, base_value + a)
    else:
        xlim = (base_value - b, base_value + b)
    # Adjust xlim to include a little visual margin.
    a = (xlim[1] - xlim[0]) * 0.02
    xlim = (xlim[0] - a, xlim[1] + a)

    return (
        base_value,
        cumsum,
        features_display,
        feature_names_display,
        xlim,
    )


def decision_plotly(
    base_value: Union[np.ndarray, float],
    shap_values: np.ndarray,
    features=None,
    feature_names=None,
    nlines=100,
    max_display=20,
    verbose=False,
    save_to=None,
):
    shap_values = shap.utils.sample(shap_values, nlines)
    features = shap.utils.sample(features, nlines)
    ret = decision_data(base_value, shap_values, features, feature_names, max_display)
    base_value, cumsum, features, feature_names, xlim = ret
    feature_names += ["Total"]
    colors = get_colors(cumsum[:, -1], *xlim)
    fig = go.Figure()
    y_pos = np.arange(0, max_display + 1)
    for i in range(cumsum.shape[0]):
        # f = np.round(features[i, :], 2)
        o = go.Scatter(
            x=cumsum[i, :],
            y=y_pos,
            # text=f.astype(str),
            hovertemplate="cumulative shapley value: %{x}<br>feature name: %{y}<extra></extra>",
            mode="lines",
            line=dict(color=colors[i]),
        )
        fig.add_trace(o)
    fig.add_vline(x=base_value, line=dict(width=1))
    colorbar_trace = go.Scatter(
        x=[None],
        y=[None],
        xaxis="x2",
        marker=dict(
            colorscale=RED_BLUE,
            showscale=True,
            colorbar=dict(
                y=0.99,
                tickvals=[],
                ticktext=[],
                orientation="h",
                thickness=5,
                outlinewidth=0,
            ),
        ),
    )
    fig.add_trace(colorbar_trace)
    xdict = dict(range=xlim, zeroline=False, showline=False, showgrid=False)
    fig.update_layout(
        xaxis=dict(title="Model output value", **xdict),
        xaxis2=dict(side="top", **xdict, overlaying="x", ticklabelposition="inside"),
        hovermode="closest",
        showlegend=False,
        yaxis=dict(
            tickvals=list(range(len(feature_names))),
            ticktext=feature_names,
            range=(0, max_display),
            zeroline=False,
            showline=False,
            showgrid=True,
        ),
        # paper_bgcolor="white",
        # plot_bgcolor="white",
    )

    if verbose:
        fig.show()
    if save_to is not None:
        with open(save_to, "w", encoding="utf8") as f:
            json.dump(fig.to_plotly_json(), f, cls=NumpyEncoder)
