import os

import tensorflow as tf  # type: ignore [import]

from zenml.pipelines import pipeline

# Path to a pip requirements file that contains requirements necessary to run
# the pipeline
requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")

@pipeline(enable_cache=True, requirements_file=requirements_file)
def continuous_deployment_pipeline(
    importer,
    normalizer,
    trainer,
    evaluator,
    drift_splitter,
    drift_detector,
    deployment_trigger,
    model_deployer,
    discord_bot
):
    # Link all the steps' artifacts together
    x_train, y_train, x_test, y_test = importer()
    x_trained_normed, x_test_normed = normalizer(x_train=x_train, x_test=x_test)
    model = trainer(x_train=x_trained_normed, y_train=y_train)
    accuracy = evaluator(x_test=x_test_normed, y_test=y_test, model=model)
    reference_dataset, new_dataset = drift_splitter(x_train)
    drift_report, _ = drift_detector(reference_dataset, new_dataset)
    deployment_decision = deployment_trigger(drift_report)
    model_deployer(deployment_decision)
    discord_bot(deployment_decision)


