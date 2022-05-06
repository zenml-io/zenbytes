import mlflow
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from zenml.steps import BaseStepConfig, step


class TrainerConfig(BaseStepConfig):
    """Trainer params"""

    epochs: int = 1
    lr: float = 0.001


@step(enable_cache=False)
def svc_trainer_mlflow(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> ClassifierMixin:
    """Train a sklearn SVC classifier and log to MLflow."""
    mlflow.sklearn.autolog()
    model = SVC(gamma=0.001)
    model.fit(X_train, y_train)
    return model


@step(enable_cache=False)
def tree_trainer_with_mlflow(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> ClassifierMixin:
    """Train a sklearn decision tree classifier and log to MLflow."""
    mlflow.sklearn.autolog()
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model
