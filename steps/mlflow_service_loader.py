from zenml.services.utils import load_last_service_from_step
from zenml.steps import step, BaseStepConfig, StepContext
from zenml.integrations.mlflow.services import MLFlowDeploymentService

class MLFlowDeploymentLoaderStepConfig(BaseStepConfig):
    """MLflow deployment getter configuration
    Attributes:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
    """

    pipeline_name: str
    step_name: str
    running: bool = True


@step(enable_cache=False)
def mlflow_service_loader(
    config: MLFlowDeploymentLoaderStepConfig, context: StepContext
) -> MLFlowDeploymentService:
    """Get the prediction service started by the deployment pipeline"""

    service = load_last_service_from_step(
        pipeline_name=config.pipeline_name,
        step_name=config.step_name,
        step_context=context,
        running=config.running,
    )
    if not service:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{config.step_name} step in the {config.pipeline_name} pipeline "
            f"is currently running."
        )

    return service
