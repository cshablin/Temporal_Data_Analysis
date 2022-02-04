import os
from joblib import load
import pandas as pd
import numpy as np
from typing import Tuple, List

from gendis.genetic import GeneticExtractor
from sklearn.metrics import classification_report


def prepare_x_y_data(mixed_df: pd.DataFrame, columns: List[str], step: int, time_window: int) -> Tuple[np.ndarray, np.ndarray]:

    mixed_label_multivariate_split_data, mixed_y = split_time_series_with_labels(mixed_df[columns],
                                                                                 mixed_df['Attack LABLE (1:No Attack, -1:Attack)'].values,
                                                                                 step, time_window)

    return mixed_label_multivariate_split_data, mixed_y


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


def extract_shapelets(column: int, col_folder: str, x_test):
    x_c_test = x_test[:, column, :].T  # shape (x_n_ts, window_length)
    genetic_extractor = GeneticExtractor.load(col_folder + os.path.sep + 'model.p')
    # construct distance matrix
    distances_test = genetic_extractor.transform(x_c_test)
    return distances_test


def evaluate(tested_df: pd.DataFrame, shapelets_folder: str, clf_path: str, training_clf_df: pd.DataFrame,
             step: int, window: int) -> Tuple[pd.DataFrame, List[int], List[int]]:
    """

    :param tested_df:
    :param shapelets_folder:
    :param clf_path:
    :param training_clf_df:
    :param step:
    :param window:
    :return:
    """
    x, y = prepare_x_y_data(tested_df, list(tested_df.columns), step, window)
    columns_path = shapelets_folder + os.path.sep + 'columns.npy'
    ordered_columns = np.load(columns_path)
    x_multi_var_distances_test = None
    clf = load(clf_path)
    for i_col in range(len(ordered_columns)):
        col_str = ordered_columns[i_col]
        column_folder = shapelets_folder + os.path.sep + col_str
        col_index_in_tested_df = tested_df.columns.get_loc(col_str)
        distances_c_test = extract_shapelets(col_index_in_tested_df, column_folder, x)
        if i_col == 0:
            x_multi_var_distances_test = distances_c_test
            continue
        x_multi_var_distances_test = np.concatenate((x_multi_var_distances_test, distances_c_test), axis=1)

    predicted = clf.predict(x_multi_var_distances_test)
    rep = classification_report(y, predicted, target_names=['attack', 'normal'])
    print('report = \n{}'.format(rep))
    return pd.DataFrame(data=x_multi_var_distances_test, columns=list(training_clf_df.columns)), list(predicted), list(y)


step = 25
window = 750
path_to_shapelets = '..\\test_configuration_3_win_500'
clf_path = 'GridSearchCV_best_lg_86.joblib'
input_df = pd.read_csv('NewShapeletsDemoTest.csv')
training_clf_df = pd.read_csv('training_lg_clf_df.csv')
df, predicted, actual = evaluate(input_df, path_to_shapelets, clf_path, training_clf_df, step, window) # input_df[880:1635]

df.to_csv("demoDF.csv", index = False)
pred_actual = {"predicted":predicted, "actual":actual}
outdf = pd.DataFrame.from_dict(pred_actual)
outdf.to_csv("pred_and_actual.csv", index = False)

