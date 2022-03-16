import pandas as pd
from typing import Dict, List, Tuple

from datetime import date, timedelta
from zenml.steps.step_output import Output
from zenml.steps import step
from zenml.steps.base_step_config import BaseStepConfig


class TrainingSplitConfig(BaseStepConfig):
    """Split config for reference_data_splitter.
    
    Attributes:
        new_data_split_date: Date to split on.
        start_reference_time_frame: Reference time to start from.
        end_reference_time_frame: Reference time to end on.
        columns: optional list of column names to use, empty means all.
    """
    new_data_split_date: str
    start_reference_time_frame: str
    end_reference_time_frame: str
    columns: List = []



@step
def reference_data_splitter(
    dataset: pd.DataFrame, config: TrainingSplitConfig
) -> Output(before=pd.DataFrame, after=pd.DataFrame):
    """Splits data for drift detection."""
    cols = config.columns if config.columns else dataset.columns
    dataset["GAME_DATE"] = pd.to_datetime(dataset["GAME_DATE"])
    dataset.set_index("GAME_DATE")

    reference_dataset = dataset.loc[
        dataset["GAME_DATE"].between(
            config.start_reference_time_frame,
            config.end_reference_time_frame,
            inclusive=True,
        )
    ][cols]

    print(reference_dataset.shape[0])

    new_data = dataset[dataset["GAME_DATE"] >= config.new_data_split_date][
        cols
    ]

    return reference_dataset, new_data
