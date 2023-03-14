import json
from typing import Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import shap
from shap.utils import approximate_interactions, convert_name

from .utils import NumpyEncoder


def dependence_plotly(
    ind: Union[int, str],
    shap_value: shap.Explanation,
    interaction_index="auto",
    save_to: Optional[str] = None,
    verbose: bool = False,
):
    features = (
        shap_value.display_data
        if shap_value.display_data is not None
        else shap_value.data
    )
    feature_names = shap_value.feature_names
    values = shap_value.values

    # fallback feature names
    if feature_names is None:
        feature_names = np.array([f"Feature {i}" for i in range(len(values))])

    ind = convert_name(ind, values, feature_names)
    if interaction_index == "auto":
        interaction_index = approximate_interactions(ind, values, features)[0]
    interaction_index = convert_name(interaction_index, values, feature_names)
    # print(ind, interaction_index)
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

    fig = px.scatter(df, x=keys[0], y=keys[1], color=keys[2])
    if verbose:
        fig.show()
    if save_to:
        with open(save_to, "w", encoding="utf8") as f:
            json.dump(fig.to_plotly_json(), f, cls=NumpyEncoder)
