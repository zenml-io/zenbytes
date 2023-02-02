import mlflow
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from zenml.steps import step


@step(enable_cache=False, experiment_tracker="mlflow_tracker")
def svc_trainer_mlflow(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> ClassifierMixin:
    """Train an sklearn SVC classifier and log to MLflow."""
    mlflow.sklearn.autolog()  # log all model hparams and metrics to MLflow
    model = SVC(gamma=0.001)
    model.fit(X_train, y_train)
    return model


@step(enable_cache=False, experiment_tracker="mlflow_tracker")
def tree_trainer_with_mlflow(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> ClassifierMixin:
    """Train an sklearn decision tree classifier and log to MLflow."""
    mlflow.sklearn.autolog()  # log all model hparams and metrics to MLflow
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model
