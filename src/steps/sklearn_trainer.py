"""Steps for training ML models."""

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from zenml.steps import step


@step
def svc_trainer(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> ClassifierMixin:
    """Train a sklearn SVC classifier."""
    model = SVC(gamma=0.001)
    model.fit(X_train, y_train)
    return model


@step()
def tree_trainer(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> ClassifierMixin:
    """Train a sklearn decision tree classifier."""
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model
