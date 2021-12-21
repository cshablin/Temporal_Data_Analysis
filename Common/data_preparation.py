import pandas as pd
import numpy as np
from typing import Tuple


def crop_array(arr: np.ndarray, start: int, stop: int) -> np.ndarray:
    return arr[start:stop]


def split_time_series_data(column: pd.Series, step: int = 500, sample_size: int = 1000) -> np.ndarray:
    # note the attack takes ~ 700 seconds (time stamps)
    np_arr_col = column.values
    chunks = []
    last_index = np_arr_col.size - 1
    for index in range(0, last_index, step):
        if index + sample_size > last_index:
            break
        chunks.append(crop_array(np_arr_col, index, index + sample_size))
    stacked_arrays = np.stack(chunks, axis=0)
    return stacked_arrays


def split_time_series_with_labels(columns_4_split: pd.DataFrame, labels: np.ndarray, step: int = 500, sample_size: int = 1000) \
        -> Tuple[np.ndarray, np.ndarray]:
    '''

    :param columns_4_split: shaped (n_timestamps, n_columns)
    :param labels: column with labels
    :param step: step size in timestamps between adjacent samples
    :param sample_size: also sliding window size
    :return: x_matrix shaped (~ n_timestamps\\step, sample_size, n_columns), y_vector shaped (~ n_timestamps\\step, 1)
    '''

    # TODO: smart usage of the scarce negative samples, we can do small steps when arriving to negative sample. This way
    #  we will produce more negative sample in result
    df_np_arr = columns_4_split.values
    return df_np_arr, labels
