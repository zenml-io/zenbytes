import os

import mlflow  # type: ignore [import]
import numpy as np  # type: ignore [import]
from sklearn.base import ClassifierMixin

from zenml.integrations.mlflow.mlflow_step_decorator import enable_mlflow
from zenml.steps import step, Output, BaseStepConfig

# Define the step and enable MLflow (n.b. order of decorators is important here)


@step
def evaluator(
    X_test: np.ndarray,
    y_test: np.ndarray,
    model: ClassifierMixin,
) -> float:
    """Calculate the accuracy on the test set"""
    test_acc = model.score(X_test, y_test)
    print(f"Test accuracy: {test_acc}")
    return test_acc
