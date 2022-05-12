from zenml.steps import step


@step
def deployment_trigger(val_acc: float) -> bool:
    """Only deploy if the validation accuracy > 90%."""
    return val_acc > 0.9
