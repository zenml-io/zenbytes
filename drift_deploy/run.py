#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.

from steps.deployment_trigger import deployment_trigger
from steps.discord_bot import discord_alert
from steps.evaluator import tf_evaluator
from steps.importer import importer_mnist
from steps.normailzer import normalizer
from steps.trainer import TrainerConfig, tf_trainer  # type: ignore [import]

from zenml.integrations.mlflow.steps import mlflow_deployer_step
from zenml.services import load_last_service_from_step
from steps.splitter import reference_data_splitter, TrainingSplitConfig
from pipelines.training_pipeline import continuous_deployment_pipeline
from zenml.integrations.mlflow.steps import MLFlowDeployerConfig

from zenml.integrations.evidently.steps import (
    EvidentlyProfileConfig,
    EvidentlyProfileStep,
)

drift_data_split_config = TrainingSplitConfig(
    row=30000,
    add_noise=True)

evidently_profile_config = EvidentlyProfileConfig(
    column_mapping=None,
    profile_sections=["datadrift"])

model_deployer = mlflow_deployer_step(name="model_deployer")

def main(epochs: int = 5, lr: float = 0.003, min_accuracy: float = 0.92, stop_service: bool = True):

    if stop_service:
        service = load_last_service_from_step(
            pipeline_name="continuous_deployment_pipeline",
            step_name="model_deployer",
            running=True,
        )
        if service:
            service.stop(timeout=10)
        return

    # Initialize a continuous deployment pipeline run
    deployment = continuous_deployment_pipeline(
        importer=importer_mnist(),
        normalizer=normalizer(),
        trainer=tf_trainer(config=TrainerConfig(epochs=epochs, lr=lr)),
        evaluator=tf_evaluator(),
        drift_splitter=reference_data_splitter(drift_data_split_config),
        drift_detector=EvidentlyProfileStep(evidently_profile_config),
        deployment_trigger=deployment_trigger(),
        model_deployer=model_deployer(config=MLFlowDeployerConfig(workers=3)),
        discord_bot=discord_alert()
    )

    deployment.run()