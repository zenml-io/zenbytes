import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin

from sklearn.base import ClassifierMixin
from zenml.steps import step, Output, BaseStepConfig
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from zenml.pipelines import pipeline
from zenml.steps import Output, step


class TrainerConfig(BaseStepConfig):
    """Trainer params"""

    epochs: int = 1
    lr: float = 0.001


# from zenml.integrations.mlflow.mlflow_step_decorator import enable_mlflow
# import mlflow


# @enable_mlflow
@step(enable_cache=False)
def svc_trainer_mlflow(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> ClassifierMixin:
    """Train another simple sklearn classifier for the digits dataset."""
    # mlflow.sklearn.autolog()
    model = SVC(gamma=0.001)
    model.fit(X_train, y_train)
    return model


# @enable_mlflow
@step(enable_cache=False)
def tree_trainer_with_mlflow(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> ClassifierMixin:
    """Train another simple sklearn classifier for the digits dataset."""
    # mlflow.sklearn.autolog()
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model
