from zenml.pipelines import pipeline

# Path to a pip requirements file that contains requirements necessary to run
# the pipeline


@pipeline(
    enable_cache=False,
    requirements_file="../requirements.txt",
    required_integrations=["seldon", "mlflow", "evidently"],
)
def inference_pipeline(
    inference_data_loader,
    prediction_service_loader,
    predictor,
):
    """Basic inference pipeline."""
    inference_data = inference_data_loader()
    model_deployment_service = prediction_service_loader()
    predictor(model_deployment_service, inference_data)
