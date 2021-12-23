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
    """

    :param columns_4_split: shaped (n_timestamps, n_columns)
    :param labels: column with labels
    :param step: step size in timestamps between adjacent samples
    :param sample_size: also sliding window size
    :return: x_matrix shaped (sample_size, n_columns, ~ n_timestamps\\step), y_vector shaped (~ n_timestamps\\step,)
    """
    df_np_arr = columns_4_split.values
    chunks = []
    result_labels = []
    last_index = df_np_arr.shape[0] - 1
    for index in range(0, last_index, step):
        if index + sample_size > last_index:
            break
        to_index = index + sample_size
        chunks.append(df_np_arr[index: to_index, :])
        result_labels.append(labels[to_index])
    stacked_arrays = np.stack(chunks, axis=2)
    return stacked_arrays, np.array(result_labels)


def split_time_series_with_negative_labels(columns_4_split: pd.DataFrame,
                                           labels: np.ndarray, step: int = 500, sample_size: int = 1000,
                                           min_negative_last_chunk_size: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param columns_4_split: shaped (n_timestamps, n_columns)
    :param labels: column with labels
    :param step: step size in timestamps between adjacent samples
    :param sample_size: also sliding window size
    :param min_negative_last_chunk_size: num of negative timestamps in succession at the end of returned samples
    :return: x_matrix shaped (sample_size, n_columns, ~ n_timestamps\\step), y_vector shaped (~ n_timestamps\\step,)
    """
    # import collections
    # smart usage of the scarce negative samples, do small steps when arriving to negative sample. This way
    # we will produce more negative sample in result
    # Counter({1: 162824, -1: 9977})
    negative_indexes = np.where(labels == -1)[0]

    df_np_arr = columns_4_split.values
    chunks = []
    result_labels = []
    # last_index = df_np_arr.shape[0] - 1
    for i in range(0, negative_indexes.shape[0], step):
        # check if previous indices are successors - fast hack
        cur_neg_index = negative_indexes[i]
        if cur_neg_index - negative_indexes[i - min_negative_last_chunk_size] == min_negative_last_chunk_size:
            chunks.append(df_np_arr[cur_neg_index - sample_size: cur_neg_index, :])
            result_labels.append(-1)

        # temp_neg_index = cur_neg_index
        # succession = True
        # for j in range(min_negative_last_chunk_size - 1):
        #     prev_neg_index = negative_indexes[i - (j + 1)]
        #     if temp_neg_index - prev_neg_index != 1:
        #         succession = False
        #         break
        #     temp_neg_index = prev_neg_index
        # if succession:
        #     chunks.append(df_np_arr[cur_neg_index - sample_size: cur_neg_index, :])
        #     result_labels.append(-1)
    # for i in range(0, negative_indexes.shape[0], min_negative_last_chunk_size):
    #     if i % min_negative_last_chunk_size == min_negative_last_chunk_size - 1:
    #         neg_index = negative_indexes[i]
    #         chunks.append(df_np_arr[neg_index - sample_size: neg_index, :])
    #         result_labels.append(-1)
    stacked_arrays = np.stack(chunks, axis=2)
    return stacked_arrays, np.array(result_labels)