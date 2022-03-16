import os

import mlflow  # type: ignore [import]
import numpy as np  # type: ignore [import]
import tensorflow as tf  # type: ignore [import]

from zenml.integrations.mlflow.mlflow_step_decorator import enable_mlflow
from zenml.steps import step
# Define the step and enable MLflow (n.b. order of decorators is important here)


@enable_mlflow
@step
def tf_evaluator(
    x_test: np.ndarray,
    y_test: np.ndarray,
    model: tf.keras.Model,
) -> float:
    """Calculate the loss for the model for each epoch in a graph"""

    _, test_acc = model.evaluate(x_test, y_test, verbose=2)
    mlflow.log_metric("val_accuracy", test_acc)
    return test_acc
    