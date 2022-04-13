import numpy as np
from zenml.steps import Output, step

def get_data_from_api():
    data = np.array([[ 0.,  0.,  1., 11., 14., 15.,  3.,  0.,  0.,  1., 13., 16., 12.,
        16.,  8.,  0.,  0.,  8., 16.,  4.,  6., 16.,  5.,  0.,  0.,  5.,
        15., 11., 13., 14.,  0.,  0.,  0.,  0.,  2., 12., 16., 13.,  0.,
         0.,  0.,  0.,  0., 13., 16., 16.,  6.,  0.,  0.,  0.,  0., 16.,
        16., 16.,  7.,  0.,  0.,  0.,  0., 11., 13., 12.,  1.,  0.]])
    # data = np.array([x.reshape(1, 8, 8) for x in data])
    return data


@step(enable_cache=False)
def dynamic_importer() -> Output(data=np.ndarray):
    """Downloads the latest data from a mock API."""
    data = get_data_from_api()
    return data
