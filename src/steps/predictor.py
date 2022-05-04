import numpy as np

from zenml.steps import step, Output
from zenml.services import BaseService


@step
def predictor(
    service: BaseService,
    data: np.ndarray,
) -> Output(predictions=np.ndarray):
    """Run a inference request against a prediction service"""

    service.start(timeout=120)  # should be a NOP if already started
    prediction = service.predict(data)
    # prediction = prediction.argmax(axis=-1)
    print("Prediction: ", prediction)
    return prediction
