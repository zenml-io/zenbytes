from zenml.pipelines import pipeline

# Path to a pip requirements file that contains requirements necessary to run
# the pipeline


@pipeline(
    enable_cache=False,
    requirements_file="../requirements.txt",
    required_integrations=["seldon", "mlflow", "evidently"],
)
def inference_pipeline(
    dynamic_importer,
    prediction_service_loader,
    predictor,
):
    """Create inference pipeline"""
    # Link all the steps artifacts together
    batch_data = dynamic_importer()
    model_deployment_service = prediction_service_loader()
    predictor(model_deployment_service, batch_data)
