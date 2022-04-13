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
import click
from rich import print
from datetime import datetime

import pipelines.training_pipeline

from pipelines.training_pipeline import continuous_deployment_pipeline
from pipelines.inference_pipeline import inference_pipeline
from steps.deployment_trigger import deployment_trigger
from steps.discord_bot import discord_alert
from steps.dynamic_importer import dynamic_importer
from steps.evaluator import evaluator
from steps.importer import importer, get_reference_data
from steps.predictor import predictor
from steps.seldon_service_loader import (
    seldon_service_loader,
    SeldonDeploymentLoaderStepConfig,
)
from steps.trainer import svc_trainer_mlflow  # type: ignore [import]


from zenml.pipelines import Schedule
from zenml.repository import Repository
from zenml.services import load_last_service_from_step

from zenml.integrations.evidently.steps import (
    EvidentlyProfileConfig,
    EvidentlyProfileStep,
)

from zenml.integrations.seldon.model_deployers import SeldonModelDeployer
from zenml.integrations.seldon.services import (
    SeldonDeploymentConfig,
    SeldonDeploymentService,
)
from zenml.integrations.seldon.steps import (
    SeldonDeployerStepConfig,
    seldon_model_deployer_step,
)


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
@click.option(
    "--stop-service",
    is_flag=True,
    default=False,
    help="Stop the MLflow prediction service",
)
def main(
    deploy: bool,
    predict: bool,
    interval_second: int,
    secret: str,
    stop_service: bool,
):
    """Run the example continuous deployment or inference pipeline
    Example usage:

        python run.py --deploy --predict --min-accuracy 0.80 \
            --secret seldon-init-container-secret

    """

    # detect the active model deployer and use Seldon Core or MLflow
    # depending on what's available
    model_deployer = Repository().active_stack.model_deployer
    use_seldon = model_deployer and isinstance(model_deployer, SeldonModelDeployer)
    pipelines.training_pipeline.DEPLOYER_TAKES_IN_MODEL = use_seldon

    if stop_service and not use_seldon:
        service = load_last_service_from_step(
            pipeline_name="continuous_deployment_pipeline",
            step_name="model_deployer",
            running=True,
        )
        if service:
            service.stop(timeout=10)
        return

    evidently_profile_config = EvidentlyProfileConfig(
        column_mapping=None, profile_sections=["datadrift"]
    )

    if deploy:

        if use_seldon:
            model_deployer_step = seldon_model_deployer_step(
                config=SeldonDeployerStepConfig(
                    service_config=SeldonDeploymentConfig(
                        model_name="mnist",
                        replicas=1,
                        implementation="SKLEARN_SERVER",
                        secret_name=secret,
                    ),
                    timeout=120,
                )
            )
        else:
            from zenml.integrations.mlflow.steps import (
                mlflow_deployer_step,
                MLFlowDeployerConfig,
            )

            model_deployer_step = mlflow_deployer_step(name="model_deployer")(
                config=MLFlowDeployerConfig(workers=1)
            )

        # Initialize a continuous deployment pipeline run
        deployment = continuous_deployment_pipeline(
            importer=importer(),
            trainer=svc_trainer_mlflow(),
            evaluator=evaluator(),
            # EvidentlyProfileStep takes reference_dataset and comparison dataset
            get_reference_data=get_reference_data(),
            drift_detector=EvidentlyProfileStep(config=evidently_profile_config),
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

        if use_seldon:
            service_loader_step = seldon_service_loader(
                config=SeldonDeploymentLoaderStepConfig(
                    pipeline_name="continuous_deployment_pipeline",
                    step_name="seldon_model_deployer_step",
                    model_name="mnist",
                )
            )
        else:
            from steps.mlflow_service_loader import (
                mlflow_service_loader,
                MLFlowDeploymentLoaderStepConfig,
            )

            service_loader_step = mlflow_service_loader(
                MLFlowDeploymentLoaderStepConfig(
                    pipeline_name="continuous_deployment_pipeline",
                    step_name="model_deployer",
                )
            )

        # Initialize an inference pipeline run
        inference = inference_pipeline(
            dynamic_importer=dynamic_importer(),
            prediction_service_loader=service_loader_step,
            predictor=predictor(),
        )

        inference.run()

    if use_seldon:
        services = model_deployer.find_model_server(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="seldon_model_deployer_step",
            model_name="mnist",
        )
        if services:
            service = services[0]
            if service.is_running:
                print(
                    f"The Seldon prediction server is running remotely as a Kubernetes "
                    f"service and accepts inference requests at:\n"
                    f"    {service.prediction_url}\n"
                    f"To stop the service, re-run the same command and supply the "
                    f"`--stop-service` argument."
                )
            elif service.is_failed:
                print(
                    f"The Seldon prediction server is in a failed state:\n"
                    f" Last state: '{service.status.state.value}'\n"
                    f" Last error: '{service.status.last_error}'"
                )

        else:
            print(
                "No Seldon prediction server is currently running. The deployment "
                "pipeline must run first to train a model and deploy it. Execute "
                "the same command with the `--deploy` argument to deploy a model."
            )
    else:
        try:
            service = load_last_service_from_step(
                pipeline_name="continuous_deployment_pipeline",
                step_name="model_deployer",
                running=True,
            )
            print(
                f"The MLflow prediction server is running locally as a daemon process "
                f"and accepts inference requests at:\n"
                f"    {service.prediction_uri}\n"
                f"To stop the service, re-run the same command and supply the "
                f"`--stop-service` argument."
            )
        except KeyError:
            print(
                "No MLflow prediction server is currently running. The deployment "
                "pipeline must run first to train a model and deploy it. Execute "
                "the same command with the `--deploy` argument to deploy a model."
            )


if __name__ == "__main__":
    main()
