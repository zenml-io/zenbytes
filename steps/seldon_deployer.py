import os
from asyncio.log import logger
from typing import Optional, cast

from pydantic import ValidationError  # type: ignore [import]
from zenml.artifacts import ModelArtifact
from zenml.environment import Environment
from zenml.integrations.constants import SELDON, SKLEARN, TENSORFLOW
from zenml.integrations.seldon.services import (SeldonDeploymentConfig,
                                                SeldonDeploymentService)
from zenml.io import fileio
from zenml.pipelines import pipeline
from zenml.services import load_last_service_from_step
from zenml.steps import BaseStepConfig, Output, StepContext, step
from zenml.steps.step_environment import STEP_ENVIRONMENT_NAME, StepEnvironment


class SeldonDeployerConfig(BaseStepConfig):
    """Seldon model deployer step configuration
    Attributes:
        model_name: the name of the Seldon model logged in the Seldon artifact
            store for the current pipeline.
        replicas: number of server replicas to use for the prediction service
        implementation: the Seldon Core implementation to use for the prediction
            service
    """

    model_name: str = "model"
    replicas: int = 1
    implementation: str
    secret_name: Optional[str]
    kubernetes_context: Optional[str]
    namespace: Optional[str]
    base_url: str
    timeout: int
    step_name: Optional[str]


@step(enable_cache=True)
def seldon_model_deployer(
    deploy_decision: bool,
    config: SeldonDeployerConfig,
    model: ModelArtifact,
    context: StepContext,
) -> SeldonDeploymentService:
    """Seldon Core model deployer pipeline step
    Args:
        deploy_decision: whether to deploy the model or not
        config: configuration for the deployer step
        model: the model artifact to deploy
        context: pipeline step context
    Returns:
        Seldon Core deployment service
    """
    # Find a service created by a previous run of this step
    step_env = cast(StepEnvironment, Environment()[STEP_ENVIRONMENT_NAME])
    step_name = config.step_name or step_env.step_name

    logger.info(
        f"Loading last service deployed by step {step_name} and "
        f"pipeline {step_env.pipeline_name}..."
    )

    try:
        service = cast(
            SeldonDeploymentService,
            # TODO [HIGH]: catch errors that are raised because a previous step used
            #   an integration that is no longer available
            load_last_service_from_step(
                pipeline_name=step_env.pipeline_name,
                step_name=step_name,
                step_context=context,
            ),
        )
    except KeyError as e:
        # pipeline or step name not found (e.g. never ran before)
        logger.error(f"No service found: {str(e)}.")
        service = None
    except ValidationError as e:
        # invalid service configuration (e.g. missing required fields because
        # the previous pipeline was run with an older service version and
        # the schemas are not compatible)
        logger.error(f"Invalide service found: {str(e)}.")
        service = None
    if service and not isinstance(service, SeldonDeploymentService):
        logger.error(
            f"Last service deployed by step {step_name} and "
            f"pipeline {step_env.pipeline_name} has invalid type. Expected "
            f"SeldonDeploymentService, found {type(service)}."
        )
        service = None

    def prepare_service_config(model_uri: str) -> SeldonDeploymentConfig:
        """Prepare the model files for model serving and create and return a
        Seldon service configuration for the model.
        This function ensures that the model files are in the correct format
        and file structure required by the Seldon Core server implementation
        used for model serving.
        Args:
            model_uri: the URI of the model artifact being served
        Returns:
            The URL to the model ready for serving.
        """
        served_model_uri = os.path.join(
            context.get_output_artifact_uri(), "seldon"
        )
        fileio.make_dirs(served_model_uri)

        # TODO [MEDIUM]: validate the model artifact type against the
        #   supported built-in Seldon server implementations
        if config.implementation == "TENSORFLOW_SERVER":
            # the TensorFlow server expects model artifacts to be
            # stored in numbered subdirectories, each representing a model version
            fileio.copy_dir(model_uri, os.path.join(served_model_uri, "1"))
        elif config.implementation == "SKLEARN_SERVER":
            # the sklearn server expects model artifacts to be
            # stored in a file called model.joblib
            model_uri = os.path.join(model.uri, "model")
            if not fileio.file_exists(model.uri):
                raise RuntimeError(
                    f"Expected sklearn model artifact was not found at {model_uri}"
                )
            fileio.copy(
                model_uri, os.path.join(served_model_uri, "model.joblib")
            )
        else:
            # TODO [MEDIUM]: implement model preprocessing for other built-in
            #   Seldon server implementations
            pass

        return SeldonDeploymentConfig(
            model_uri=served_model_uri,
            model_name=config.model_name,
            # TODO [MEDIUM]: auto-detect built-in Seldon server implementation
            #   from the model artifact type
            implementation=config.implementation,
            secret_name=config.secret_name,
            pipeline_name=step_env.pipeline_name,
            pipeline_run_id=step_env.pipeline_run_id,
            pipeline_step_name=step_name,
            replicas=config.replicas,
            kubernetes_context=config.kubernetes_context,
            namespace=config.namespace,
            base_url=config.base_url,
        )

    if not deploy_decision:
        print(
            "Skipping model deployment because the model quality does not meet "
            "the criteria"
        )
        if not service:
            service_config = prepare_service_config(model.uri)
            service = SeldonDeploymentService(config=service_config)
        return service

    service_config = prepare_service_config(model.uri)
    if service and not service.is_stopped:
        print("Updating an existing Seldon deployment service")
        service.update(service_config)
    else:
        print("Creating a new Seldon deployment service")
        service = SeldonDeploymentService(config=service_config)

    service.start(timeout=config.timeout)
    print(
        f"Seldon deployment service started and reachable at:\n"
        f"    {service.prediction_url}\n"
    )

    return service
