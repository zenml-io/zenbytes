from zenml.steps import step

@step
def deployment_trigger(
    drift_report: dict,
) -> bool:
    """Implements a simple model deployment trigger that looks at the
    drift report and deploys if there's none"""

    drift = drift_report["data_drift"]["data"]["metrics"]["dataset_drift"]

    if drift:
        return False
    else:
        return True
