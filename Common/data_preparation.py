import pandas as pd
import numpy as np


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
