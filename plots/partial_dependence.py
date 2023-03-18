import json
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap

from .utils import DEFAULT_CONFIG, DEFAULT_LAYOUT, NumpyEncoder


def partial_dependence_plotly(
    ind,
    model,
    features: np.ndarray,
    shap_values: Optional[shap.Explanation] = None,
    feature_names=None,
    npoints: int = 100,
    verbose=False,
    save_to=None,
):
    # convert from DataFrames if we got any
    use_dataframe = False
    if str(type(features)).endswith("'pandas.core.frame.DataFrame'>"):
        if feature_names is None:
            feature_names = features.columns
        features = features.values
        use_dataframe = True

    if feature_names is None:
        feature_names = ["Feature %d" % i for i in range(features.shape[1])]

    ind = shap.utils.convert_name(ind, None, feature_names)
    target_name = feature_names[ind]
    xv = features[:, ind]
    xmin, xmax = np.min(xv), np.max(xv)
    xs = np.linspace(xmin, xmax, npoints)
    features_tmp = features.copy()
    vals = np.zeros(npoints)
    for i in range(npoints):
        features_tmp[:, ind] = xs[i]
        vals[i] = model(features_tmp).mean()
        if use_dataframe:
            vals[i] = model(pd.DataFrame(features_tmp, columns=feature_names)).mean()
        else:
            vals[i] = model(features_tmp).mean()

    mval = xv.mean()
    if use_dataframe:
        model_expected_value = model(
            pd.DataFrame(features, columns=feature_names)
        ).mean()
    else:
        model_expected_value = model(features).mean()

    line_fig = go.Scatter(x=xs, y=vals, mode="lines", hoverinfo="x+y")
    hist_fig = go.Histogram(
        x=xv,
        nbinsx=50,
        opacity=0.1,
        yaxis="y2",
        hovertemplate="%{x}<br>count=%{y}<extra></extra>",
    )
    fig = go.Figure(data=[line_fig, hist_fig])
    fig.add_vline(
        x=mval,
        line_dash="dash",
        annotation_text=f"E({target_name})",
        annotation_position="top right",
    )
    fig.add_hline(
        model_expected_value,
        line_dash="dash",
        annotation_text=f"E[f({target_name})]",
        annotation_position="bottom right",
    )

    if shap_values is not None:
        assert len(shap_values.shape) == 1
        # TODO
        x0, y0 = shap_values.data[ind], model_expected_value
        x1, y1 = shap_values.data[ind], model(shap_values.data.reshape(1, -1))[0]
        print(x0, y0)
        fig.add_shape(
            type="line",
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            line=dict(color="rgb(30, 136, 229)"),
        )
        fig.add_trace(go.Scatter(x=[x1], y=[y1], mode="markers", hoverinfo="x+y"))
    fig.update_layout(
        **DEFAULT_LAYOUT,
        xaxis_title_text=target_name,
        yaxis_title_text=f"E[f(x) | {target_name}]",
        yaxis=dict(showgrid=False, autorange=True),
        yaxis2=dict(
            visible=False, overlaying="y", range=[0, features.shape[0]], showgrid=False
        ),
        showlegend=False,
    )
    if verbose:
        fig.show(config=DEFAULT_CONFIG)
    if save_to is not None:
        with open(save_to, "w", encoding="utf8") as f:
            json.dump(fig.to_plotly_json(), f, cls=NumpyEncoder)
