import lightgbm as lgb
import numpy as np
import shap
import sklearn.datasets as D

from plots import *


def load_data(return_data=False, explainer_cls=shap.Explainer):
    X, y = D.load_breast_cancer(return_X_y=True, as_frame=True)

    # explain the model's predictions using SHAP
    # (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
    model = lgb.LGBMClassifier().fit(X, y)
    explainer = explainer_cls(model)
    shap_values: shap.Explanation = explainer(X)[:, :, 0]
    if return_data:
        return model, shap_values, X
    return model, shap_values


def test_decision():
    _, shap_values = load_data()

    decision_plotly(
        shap_values.base_values,
        shap_values.values,
        shap_values.data,
        shap_values.feature_names,
        verbose=True,
        save_to="test.json",
    )
    shap.decision_plot(
        shap_values.base_values[0],
        shap_values.values,
        shap_values.data,
        shap_values.feature_names,
    )


def test_partial_dependence():
    model, shap_values = load_data(explainer_cls=shap.TreeExplainer)
    fid = np.argmax(model.feature_importances_)
    partial_dependence_plotly(
        fid,
        model.predict,
        shap_values.data,
        # shap_values=shap_values[20],
        feature_names=shap_values.feature_names,
        verbose=True,
        save_to="test.html",
    )
    shap.partial_dependence_plot(
        fid,
        model.predict,
        shap_values.data,
        model_expected_value=True,
        feature_expected_value=True,
        ice=False,
        # shap_values=shap_values[20:21, :],
        feature_names=shap_values.feature_names,
    )


def test_summary():
    _, shap_values = load_data()

    summary_plotly(
        shap_values.values,
        shap_values.data,
        shap_values.feature_names,
        verbose=True,
        save_to="test.json",
    )
    shap.summary_plot(shap_values.values, shap_values.data, shap_values.feature_names)


def test_waterfall():
    _, shap_values = load_data()
    waterfall_plotly(
        shap_values[0].values,
        shap_values[0].data,
        shap_values[0].base_values,
        shap_values[0].feature_names,
        verbose=True,
        save_to="test.json",
    )
    shap.waterfall_plot(shap_values[0])


def test_dependence():
    _, shap_values, X = load_data(True)
    dependence_plotly(
        1,
        shap_values.values,
        shap_values.data,
        shap_values.feature_names,
        verbose=True,
        save_to="test.json",
    )
    shap.dependence_plot(1, shap_values.values, X)
