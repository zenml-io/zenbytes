import numpy as np
from zenml.steps import step


@step
def inference_data_loader() -> np.ndarray:
    """Load some inference data."""
    return np.random.rand(1, 64)  # flattened 8x8 random noise image
