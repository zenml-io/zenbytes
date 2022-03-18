import numpy as np
import pandas as pd
from zenml.steps import Output, step
from zenml.integrations.sklearn.helpers.digits import (
    get_digits,
)

@step
def importer() -> Output(
    X_train=np.ndarray, X_test=np.ndarray, y_train=np.ndarray, y_test=np.ndarray
):
    """Loads the digits array as normal numpy arrays."""
    X_train, X_test, y_train, y_test = get_digits()
    return X_train, X_test, y_train, y_test

@step
def get_reference_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> Output(reference=pd.DataFrame, comparison=pd.DataFrame):
    """Splits data for drift detection."""
    # X_train = _add_awgn(X_train)
    columns = [str(x) for x in list(range(X_train.shape[1]))]
    return pd.DataFrame(X_test, columns=columns), pd.DataFrame(X_train, columns=columns)
