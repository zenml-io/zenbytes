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

from pipelines.training_pipeline import continuous_deployment_pipeline
from steps.deployment_trigger import deployment_trigger
from steps.discord_bot import discord_alert
from steps.evaluator import evaluator
from steps.importer import importer, get_reference_data
from steps.trainer import svc_trainer_mlflow # type: ignore [import]
from steps.seldon_deployer import SeldonDeployerConfig, seldon_model_deployer
from zenml.pipelines import Schedule
from zenml.services import load_last_service_from_step

from zenml.integrations.evidently.steps import (
    EvidentlyProfileConfig,
    EvidentlyProfileStep,
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
@click.option("--interval-second", help="How long between scheudle pipelines.", type=int, default=None)
@click.option("--kubernetes-context", help="Kubernetes context to use.")
@click.option("--namespace", help="Kubernetes namespace to use.")
@click.option("--base-url", help="Seldon core ingress base URL.")
@click.option(
    "--stop-service",
    is_flag=True,
    default=False,
    help="Stop the prediction service when done",
)
def main(
    deploy: bool,
    predict: bool,
    interval_second: int,
    kubernetes_context: str,
    namespace: str,
    base_url: str,
    stop_service: bool,
):
    """Run the Seldon example continuous deployment or inference pipeline
    Example usage:
    python run.py --deploy --predict \
        --kubernetes-context=zenml-eks-sandbox \
        --namespace=kubeflow \
        --base-url=http://abb84c444c7804aa98fc8c097896479d-377673393.us-east-1.elb.amazonaws.com \
    """
    evidently_profile_config = EvidentlyProfileConfig(
        column_mapping=None,
        profile_sections=["datadrift"])
    
    if stop_service:
        service = load_last_service_from_step(
            pipeline_name="continuous_deployment_pipeline",
            step_name="model_deployer",
            running=True,
        )
        if service:
            service.stop(timeout=100)
        return

    if deploy:
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
            model_deployer=seldon_model_deployer(
                config=SeldonDeployerConfig(
                    model_name="mnist",
                    step_name="model_deployer",
                    replicas=1,
                    implementation="SKLEARN_SERVER",
                    secret_name="seldon-init-container-secret",
                    kubernetes_context=kubernetes_context,
                    namespace=namespace,
                    base_url=base_url,
                    timeout=120,
                )
            ),
        )

        if interval_second is not None:
            deployment.run(
                schedule=Schedule(start_time=datetime.now(), interval_second=interval_second)
            )
        else:
            deployment.run()
            
    if predict:
        # Coming soon
        # Initialize an inference pipeline run
        # inference = inference_pipeline(
        #     dynamic_importer=dynamic_importer(),
        #     predict_preprocessor=predict_preprocessor,
        #     prediction_service_loader=prediction_service_loader(
        #         SeldonDeploymentLoaderStepConfig(
        #             pipeline_name="continuous_deployment_pipeline",
        #             step_name="model_deployer",
        #         )
        #     ),
        #     predictor=predictor(),
        # )

        # inference.run()
        raise NotImplementedError("Predict pipeline coming soon!")

    try:
        service = load_last_service_from_step(
            pipeline_name="continuous_deployment_pipeline",
            step_name="model_deployer",
            running=True,
        )
        print(
            f"The Seldon prediction server is running remotely as a Kubernetes "
            f"service and accepts inference requests at:\n"
            f"    {service.prediction_url}\n"
            f"To stop the service, re-run the same command and supply the "
            f"`--stop-service` argument."
        )
    except KeyError:
        print(
            "No Seldon prediction server is currently running. The deployment "
            "pipeline must run first to train a model and deploy it. Execute "
            "the same command with the `--deploy` argument to deploy a model."
        )


if __name__ == "__main__":
    main()