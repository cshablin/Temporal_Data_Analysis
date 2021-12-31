import datetime
import os
from queue import Queue, Empty
from threading import Thread

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Any
from gendis.genetic import GeneticExtractor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from Common.data_preparation import split_time_series_with_labels, split_time_series_with_negative_labels
from Exp_Shapelets.Config import ShapeletsConfig


class MultiVarShapeletsExtractor:
    almost_const_columns = ['2_LS_101_AH',	'2_LS_101_AL',	'2_LS_201_AH',	'2_LS_201_AL',	'2_LS_301_AH',	'2_LS_301_AL',
                            '2_LS_401_AH',	'2_LS_401_AL',	'2_LS_501_AH',	'2_LS_501_AL',	'2_LS_601_AH', '2_LS_601_AL',
                            '2_PIC_003_SP', '3_AIT_001_PV', '1_MV_004_STATUS']
    invalid_columns = ['Unnamed: 0', 'Date', 'Date ', 'Time', 'Attack LABLE (1:No Attack, -1:Attack)']

    def __init__(self, conf: ShapeletsConfig, normal_labels_train_df: pd.DataFrame, mixed_labels_train_df: pd.DataFrame,
                 normal_labels_test_df: pd.DataFrame, mixed_labels_test_df: pd.DataFrame):
        self.config = conf
        self.normal_labels_train_df = normal_labels_train_df
        self.normal_labels_test_df = normal_labels_test_df
        self.mixed_labels_train_df = mixed_labels_train_df
        self.mixed_labels_test_df = mixed_labels_test_df
        self.col_jobs_q = Queue()
        self.n_threads = 1
        self.shapelets_threads = []

    def prepare_data(self, shapelets_columns: List[str]):
        # columns = list(set(list(self.normal_labels_train_df.columns)) - set(self.almost_const_columns + self.invalid_columns ))
        columns = list(set(list(shapelets_columns)) - set(self.almost_const_columns + self.invalid_columns))
        step = self.config.step
        window_length = self.config.window_length
        step4negative = self.config.step4negative
        min_negative_last_chunk_size = self.config.min_negative_last_chunk_size
        x_train, y_train = self.__prepare_x_y_data(self.normal_labels_train_df, self.mixed_labels_train_df,
                                                   columns, step, window_length, step4negative, min_negative_last_chunk_size)
        x_test, y_test = self.__prepare_x_y_data(self.normal_labels_test_df, self.mixed_labels_test_df,
                                                 columns, step, window_length, step4negative, min_negative_last_chunk_size)
        self.__save(np.array(columns), 'columns.npy')
        self.__save(x_train, 'x_train.npy')
        self.__save(y_train, 'y_train.npy')
        self.__save(x_test, 'x_test.npy')
        self.__save(y_test, 'y_test.npy')
        for i in range(len(columns)):
            self.col_jobs_q.put((i, columns[i]))
        for i in range(self.n_threads):
            self.shapelets_threads.append(UniVarShapeletsExtractor(self.config, self.col_jobs_q, x_train, y_train))

    def __save(self, arr: np.ndarray, file_name):
        file_path = self.config.test_folder + os.path.sep + file_name
        if not os.path.exists(file_path):
            np.save(file_path, arr)

    def __load(self, file_name) -> np.ndarray:
        file_path = self.config.test_folder + os.path.sep + file_name
        return np.load(file_path)

    def discover_shapelets(self):
        for th in self.shapelets_threads:
            th.start()

    # this should be called after finished extracting shapelets for all columns
    def train_classifier(self):
        y_train = self.__load('y_train.npy')
        ordered_columns = self.__load('columns.npy')
        for i in range(len(ordered_columns)):
            col_str = ordered_columns[i]
            pass
            distances_train = genetic_extractor.transform(X_train)
            distances_test = genetic_extractor.transform(X_test)

        # Fit ML classifier on constructed distance matrix
        lr = LogisticRegression()
        lr.fit(distances_train, y_train)

    def __prepare_x_y_data(self, normal_df: pd.DataFrame, mixed_df: pd.DataFrame,
                           columns: List[str], step: int = 500, time_window: int = 1000,
                           step4negative: int = 5, min_negative_last_chunk_size: int = 20):
        negative_label_multivariate_split_data, neg_y = split_time_series_with_negative_labels(mixed_df[columns],
                                                                                               mixed_df['Attack LABLE (1:No Attack, -1:Attack)'].values,
                                                                                               step4negative, time_window, min_negative_last_chunk_size)

        # note we can't do split_time_series_with_labels() on combined DF of both dataset as they aren't chronological.
        mixed_label_multivariate_split_data, mixed_y = split_time_series_with_labels(mixed_df[columns],
                                                                                     mixed_df['Attack LABLE (1:No Attack, -1:Attack)'].values,
                                                                                     step, time_window)
        norm_label_multivariate_split_data, norm_y = split_time_series_with_labels(normal_df[columns],
                                                                                   np.ones(normal_df.shape[0]),
                                                                                   step, time_window)
        # number of samples is the third dimension
        all_multivariate_prepared_x = np.concatenate((norm_label_multivariate_split_data,
                                                      mixed_label_multivariate_split_data,
                                                      negative_label_multivariate_split_data), axis=2)
        all_multivariate_y = np.concatenate((norm_y, mixed_y, neg_y), axis=0)
        # manual shuffle
        np.random.seed(123)
        permutation = np.random.permutation(all_multivariate_prepared_x.shape[2])
        np.take(all_multivariate_prepared_x, permutation, axis=2, out=all_multivariate_prepared_x)
        np.take(all_multivariate_y, permutation, axis=0, out=all_multivariate_y)
        return all_multivariate_prepared_x, all_multivariate_y


class UniVarShapeletsExtractor(Thread):
    def __init__(self, config: ShapeletsConfig, queue: Queue, x_train: np.ndarray, y_train: np.ndarray):
        super(UniVarShapeletsExtractor, self).__init__()
        self.config = config
        self.queue = queue
        self.x_train = x_train
        self.y_train = y_train

    def run(self) -> None:
        while not self.queue.empty():
            try:
                column_job = self.queue.get_nowait()
                print(str(datetime.datetime.now()) + " start  column '{0}'".format(column_job[1]))
                col_folder = self.config.test_folder + os.path.sep + column_job[1]
                if os.path.exists(col_folder):
                    print("Skipping handling column {0}".format(column_job[1]))
                    continue
                os.makedirs(col_folder)
                self.extract_shapelets(column_job[0], col_folder)
                print(str(datetime.datetime.now()) + " Done   column '{0}'".format(column_job[1]))
            except Empty:
                pass

    def extract_shapelets(self, column: int, col_folder: str):
        x_c_train = self.x_train[:, column, :].T  # shape (x_n_ts, window_length)
        # Fit the GeneticExtractor and construct distance matrix
        genetic_extractor = GeneticExtractor(population_size=self.config.population_size, iterations=self.config.iterations,
                                             verbose=self.config.verbose, n_jobs=1, mutation_prob=self.config.mutation_prob,
                                             crossover_prob=self.config.crossover_prob, normed=self.config.normed,
                                             wait=self.config.wait, max_len=len(x_c_train) // 2, location=True)

        genetic_extractor.fit(x_c_train, self.y_train)
        distances_train = genetic_extractor.transform(x_c_train)
        np.save(col_folder + os.path.sep + 'distances_train', distances_train)
        genetic_extractor.save(col_folder + os.path.sep + 'model.p')
        # new_extractor = GeneticExtractor.load('temp.p')


# # Read in the datafiles
# train_df = pd.read_csv('<DATA_FILE>')
# test_df = pd.read_csv('<DATA_FILE>')
# # Split into feature matrices and label vectors
# X_train = train_df.drop('target', axis=1)
# y_train = train_df['target']
# X_test = test_df.drop('target', axis=1)
# y_test = test_df['target']
#
# # Univariate time series example
# # Creating a GeneticExtractor object
# genetic_extractor = GeneticExtractor(population_size=50, iterations=25, verbose=True,
#                                      mutation_prob=0.3, crossover_prob=0.3,
#                                      wait=10, max_len=len(X_train) // 2)
#
# # Fit the GeneticExtractor and construct distance matrix
# shapelets = genetic_extractor.fit(X_train, y_train)
# distances_train = genetic_extractor.transform(X_train)
# distances_test = genetic_extractor.transform(X_test)
#
# # Fit ML classifier on constructed distance matrix
# lr = LogisticRegression()
# lr.fit(distances_train, y_train)
#
# print('Accuracy = {}'.format(accuracy_score(y_test, lr.predict(distances_test))))



