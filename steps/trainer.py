import mlflow
import numpy as np
from sklearn.base import ClassifierMixin

from sklearn.base import ClassifierMixin
from zenml.steps import step, BaseStepConfig
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from zenml.steps import step


class TrainerConfig(BaseStepConfig):
    """Trainer params"""

    epochs: int = 1
    lr: float = 0.001


@step(enable_cache=False)
def svc_trainer_mlflow(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> ClassifierMixin:
    """Train another simple sklearn classifier for the digits dataset."""
    mlflow.sklearn.autolog()
    model = SVC(gamma=0.001)
    model.fit(X_train, y_train)
    return model


@step(enable_cache=False)
def tree_trainer_with_mlflow(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> ClassifierMixin:
    """Train another simple sklearn classifier for the digits dataset."""
    mlflow.sklearn.autolog()
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model
