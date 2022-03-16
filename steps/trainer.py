import os

import mlflow  # type: ignore [import]
import numpy as np  # type: ignore [import]
import pandas as pd  # type: ignore [import]
import requests  # type: ignore [import]
import tensorflow as tf  # type: ignore [import]

from zenml.integrations.mlflow.mlflow_step_decorator import enable_mlflow
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_deployer_step
from zenml.pipelines import pipeline
from zenml.services import load_last_service_from_step
from zenml.steps import BaseStepConfig, Output, StepContext, step

class TrainerConfig(BaseStepConfig):
    """Trainer params"""

    epochs: int = 1
    lr: float = 0.001


# Define the step and enable MLflow (n.b. order of decorators is important here)
@enable_mlflow
@step
def tf_trainer(
    config: TrainerConfig,
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> tf.keras.Model:
    """Train a neural net from scratch to recognize MNIST digits return our
    model or the learner"""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(10),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    mlflow.tensorflow.autolog()
    model.fit(
        x_train,
        y_train,
        epochs=config.epochs,
    )

    # write model
    return model