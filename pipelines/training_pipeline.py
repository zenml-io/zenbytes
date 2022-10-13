from zenml.pipelines import pipeline


@pipeline(enable_cache=False)
def train_evaluate_deploy_pipeline(
    importer,
    trainer,
    evaluator,
    deployment_trigger,
    model_deployer,
):
    """Train and deploy a model with MLflow."""
    X_train, X_test, y_train, y_test = importer()
    model = trainer(X_train=X_train, y_train=y_train)
    test_acc = evaluator(X_test=X_test, y_test=y_test, model=model)
    deployment_decision = deployment_trigger(test_acc)  # new
    model_deployer(deployment_decision, model)  # new
