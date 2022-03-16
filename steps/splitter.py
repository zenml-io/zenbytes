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

@step
def reference_data_splitter(
    dataset: pd.DataFrame, config: TrainingSplitConfig
) -> Output(before=pd.DataFrame, after=pd.DataFrame):
    """Splits data for drift detection."""

    reference_dataset = dataset[1:config.row]
    print(reference_dataset.shape[0])

    new_data = dataset[config.row:]

    return reference_dataset, new_data
