import pandas as pd

from zenml.steps.step_output import Output
from zenml.steps import step
from zenml.steps.base_step_config import BaseStepConfig


class TrainingSplitConfig(BaseStepConfig):
    """Split config for reference_data_splitter.
    
    Attributes:
        row: the row number of the image to split the dataset on. Value has 
            to be less than 60,000.
    """
    row: int
    add_noise: bool

@step
def reference_data_splitter(
    dataset: pd.DataFrame, config: TrainingSplitConfig
) -> Output(before=pd.DataFrame, after=pd.DataFrame):
    """Splits data for drift detection."""

    reference_dataset = dataset[1:config.row]
    print(reference_dataset.shape[0])

    new_data = dataset[config.row:]

    if config.add_noise:
        new_data = _add_awgn(new_data)

    return reference_dataset, new_data

def _add_awgn(dataset: pd.DataFrame):
    import numpy as np 
    mu, sigma = 0, 0.1 
    # creating a noise with the same dimension as the dataset
    noise = np.random.normal(mu, sigma, dataset.shape)
    print(noise)

    new_data = dataset + noise
    return new_data
