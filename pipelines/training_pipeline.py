from zenml.pipelines import pipeline

DEPLOYER_TAKES_IN_MODEL=True


@pipeline(
    enable_cache=False,
    requirements_file="../requirements.txt",
    required_integrations=["seldon", "mlflow", "evidently"],
)
def continuous_deployment_pipeline(
    importer,
    trainer,
    evaluator,
    get_reference_data,
    drift_detector,
    alerter,
    deployment_trigger,
    model_deployer,
):
    """Links all the steps together in a pipeline"""
    X_train, X_test, y_train, y_test = importer()
    model = trainer(X_train=X_train, y_train=y_train)
    evaluator(X_test=X_test, y_test=y_test, model=model)

    reference, comparison = get_reference_data(X_train, X_test)
    drift_report, _ = drift_detector(reference, comparison)

    alerter(drift_report)

    # new
    deployment_decision = deployment_trigger(drift_report)
    if DEPLOYER_TAKES_IN_MODEL:
        model_deployer(deployment_decision, model)
    else:
        model_deployer(deployment_decision)
