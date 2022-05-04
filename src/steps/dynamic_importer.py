import numpy as np
from zenml.steps import Output, step


def get_data_from_api():
    data = np.array(
        [
            [
                0.0,
                0.0,
                1.0,
                11.0,
                14.0,
                15.0,
                3.0,
                0.0,
                0.0,
                1.0,
                13.0,
                16.0,
                12.0,
                16.0,
                8.0,
                0.0,
                0.0,
                8.0,
                16.0,
                4.0,
                6.0,
                16.0,
                5.0,
                0.0,
                0.0,
                5.0,
                15.0,
                11.0,
                13.0,
                14.0,
                0.0,
                0.0,
                0.0,
                0.0,
                2.0,
                12.0,
                16.0,
                13.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                13.0,
                16.0,
                16.0,
                6.0,
                0.0,
                0.0,
                0.0,
                0.0,
                16.0,
                16.0,
                16.0,
                7.0,
                0.0,
                0.0,
                0.0,
                0.0,
                11.0,
                13.0,
                12.0,
                1.0,
                0.0,
            ]
        ]
    )
    # data = np.array([x.reshape(1, 8, 8) for x in data])
    return data


@step(enable_cache=False)
def dynamic_importer() -> Output(data=np.ndarray):
    """Downloads the latest data from a mock API."""
    data = get_data_from_api()
    return data
