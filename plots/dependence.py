import json
from typing import Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
from shap.utils import approximate_interactions, convert_name

from .utils import DEFAULT_CONFIG, DEFAULT_LAYOUT, NumpyEncoder


def dependence_plotly(
    ind: Union[int, str],
    values,
    features,
    feature_names=None,
    interaction_index="auto",
    save_to: Optional[str] = None,
    verbose: bool = False,
):
    # fallback feature names
    if feature_names is None:
        feature_names = np.array([f"Feature {i}" for i in range(len(values))])

    ind = convert_name(ind, values, feature_names)
    if interaction_index == "auto":
        interaction_index = approximate_interactions(ind, values, features)[0]
    interaction_index = convert_name(interaction_index, values, feature_names)

    keys = [
        feature_names[ind],
        f"SHAP value for\n{feature_names[ind]}",
        feature_names[interaction_index],
    ]
    df = pd.DataFrame(
        {
            keys[0]: features[:, ind],
            keys[1]: values[:, ind],
            keys[2]: features[:, interaction_index],
        }
    )

    q_5 = df[keys[2]].quantile(0.05)
    q_95 = df[keys[2]].quantile(0.95)
    df[keys[2]][df[keys[2]] < q_5] = q_5
    df[keys[2]][df[keys[2]] > q_95] = q_95

    fig = px.scatter(
        df,
        x=keys[0],
        y=keys[1],
        color=keys[2],
    )
    fig.update_layout(**DEFAULT_LAYOUT)
    if verbose:
        fig.show(config=DEFAULT_CONFIG)
    if save_to:
        with open(save_to, "w", encoding="utf8") as f:
            json.dump(fig.to_plotly_json(), f, cls=NumpyEncoder)
