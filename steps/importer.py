import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from zenml.steps import Output, step


@step
def importer() -> Output(
    X_train=np.ndarray, X_test=np.ndarray, y_train=np.ndarray, y_test=np.ndarray
):
    """Loads the digits array as normal numpy arrays."""
    digits = load_digits()
    data = digits.images.reshape((len(digits.images), -1))
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.2, shuffle=False
    )
    return X_train, X_test, y_train, y_test


@step
def get_reference_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> Output(reference=pd.DataFrame, comparison=pd.DataFrame):
    """Splits data for drift detection."""
    # X_train = _add_awgn(X_train)
    columns = [str(x) for x in list(range(X_train.shape[1]))]
    return pd.DataFrame(X_test, columns=columns), pd.DataFrame(
        X_train, columns=columns
    )
