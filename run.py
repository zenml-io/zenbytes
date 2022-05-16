#  Copyright (c) ZenML GmbH 2022. All Rights Reserved.
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
from datetime import datetime

import click
from rich import print
from zenml.integrations.evidently.steps import (
    EvidentlyProfileConfig,
    EvidentlyProfileStep,
)
from zenml.integrations.mlflow.steps import (
    MLFlowDeployerConfig,
    mlflow_model_deployer_step,
)
from zenml.integrations.seldon.model_deployers import SeldonModelDeployer
from zenml.integrations.seldon.services import SeldonDeploymentConfig
from zenml.integrations.seldon.steps import (
    SeldonDeployerStepConfig,
    seldon_model_deployer_step,
)
from zenml.pipelines import Schedule
from zenml.repository import Repository

from pipelines.inference_pipeline import inference_pipeline
from pipelines.training_pipeline import continuous_deployment_pipeline
from steps.deployment_trigger import deployment_trigger
from steps.discord_bot import discord_alert
from steps.evaluator import evaluator
from steps.importer import get_reference_data, importer
from steps.inference_data_loader import inference_data_loader
from steps.prediction_service_loader import (
    PredictionServiceLoaderStepConfig,
    prediction_service_loader,
)
from steps.predictor import predictor
from steps.sklearn_trainer import svc_trainer


@click.command()
@click.option(
    "--deploy",
    "-d",
    is_flag=True,
    help="Run the deployment pipeline to train and deploy a model",
)
@click.option(
    "--predict",
    "-p",
    is_flag=True,
    help="Run the inference pipeline to send a prediction request "
    "to the deployed model",
)
@click.option(
    "--interval-second",
    help="How long between schedule pipelines.",
    type=int,
    default=None,
)
@click.option(
    "--secret",
    "-x",
    type=str,
    default="seldon-init-container-secret",
    help="Specify the name of a Kubernetes secret to be passed to Seldon Core "
    "deployments to authenticate to the Artifact Store",
)
def main(
    deploy: bool,
    predict: bool,
    interval_second: int,
    secret: str,
):
    """Run the example continuous deployment or inference pipeline
    Example usage:

        python run.py --deploy --predict --min-accuracy 0.80 \
            --secret seldon-init-container-secret

    """

    # detect the active model deployer and use Seldon Core or MLflow
    # depending on what's available
    model_deployer = Repository().active_stack.model_deployer
    if not model_deployer:
        raise RuntimeError(
            "A Model Deployer must be configured in the active stack."
        )

    deployment_pipeline_name = "continuous_deployment_pipeline"
    if model_deployer and isinstance(model_deployer, SeldonModelDeployer):
        use_seldon = True
        deployment_step_name = "seldon_model_deployer_step"
        model_name = "mnist"
    else:
        use_seldon = False
        deployment_step_name = "mlflow_model_deployer_step"
        model_name = "model"

    evidently_profile_config = EvidentlyProfileConfig(
        column_mapping=None, profile_sections=["datadrift"]
    )

    if deploy:

        if use_seldon:
            model_trainer_step = svc_trainer
            model_deployer_step = seldon_model_deployer_step(
                config=SeldonDeployerStepConfig(
                    service_config=SeldonDeploymentConfig(
                        model_name=model_name,
                        replicas=1,
                        implementation="SKLEARN_SERVER",
                        secret_name=secret,
                    ),
                    timeout=120,
                )
            )
        else:
            model_trainer_step = svc_trainer
            model_deployer_step = mlflow_model_deployer_step(
                config=MLFlowDeployerConfig(workers=1, timeout=20)
            )

        # Initialize a continuous deployment pipeline run
        deployment = continuous_deployment_pipeline(
            importer=importer(),
            trainer=model_trainer_step(),
            evaluator=evaluator(),
            # EvidentlyProfileStep takes reference_dataset and comparison dataset
            get_reference_data=get_reference_data(),
            drift_detector=EvidentlyProfileStep(
                config=evidently_profile_config
            ),
            # Add discord
            alerter=discord_alert(),
            deployment_trigger=deployment_trigger(),
            model_deployer=model_deployer_step,
        )

        if interval_second is not None:
            deployment.run(
                schedule=Schedule(
                    start_time=datetime.now(), interval_second=interval_second
                )
            )
        else:
            deployment.run()

    if predict:

        # Initialize an inference pipeline run
        inference = inference_pipeline(
            dynamic_importer=inference_data_loader(),
            prediction_service_loader=prediction_service_loader(
                config=PredictionServiceLoaderStepConfig(
                    pipeline_name=deployment_pipeline_name,
                    step_name=deployment_step_name,
                    model_name=model_name,
                )
            ),
            predictor=predictor(),
        )

        inference.run()

    services = model_deployer.find_model_server(
        pipeline_name=deployment_pipeline_name,
        pipeline_step_name=deployment_step_name,
        model_name=model_name,
    )
    if services:
        service = services[0]
        if service.is_running:
            print(
                f"The model prediction server is running and accepts inference "
                f"requests at:\n"
                f"    {service.prediction_url}\n"
                f"To stop the service, run "
                f"[italic green]`zenml served-models delete "
                f"{str(service.uuid)}`[/italic green]."
            )
        elif service.is_failed:
            print(
                f"The model prediction server is in a failed state:\n"
                f" Last state: '{service.status.state.value}'\n"
                f" Last error: '{service.status.last_error}'"
            )

    else:
        print(
            "No model prediction server is currently running. The deployment "
            "pipeline must run first to train a model and deploy it. Execute "
            "the same command with the `--deploy` argument to deploy a model."
        )


if __name__ == "__main__":
    main()
