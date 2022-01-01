import datetime
import os
import unittest
import warnings

import numpy
import numpy as np
import pandas as pd
from gendis.genetic import GeneticExtractor
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from typing import List
from Common.data_preparation import split_time_series_data, split_time_series_with_labels, \
    split_time_series_with_negative_labels
from Exp_Shapelets.Config import ShapeletsConfig
from Exp_Shapelets.discovery_of_shapelets import MultiVarShapeletsExtractor


def ignore_warnings(test_func):
    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test_func(self, *args, **kwargs)
    return do_test


class ShapeletsTestCase(unittest.TestCase):

    def test_shapelets_in_col(self):
        '''
        columns '1_AIT_001_PV', '1_AIT_002_PV', '1_AIT_003_PV', '1_AIT_004_PV', '1_AIT_005_PV', '1_FIT_001_PV', '1_LS_001_AL',
        '''
        # print(local_df['1_AIT_001_PV'].describe())
        '''
        count    784537.000000
        mean        169.823532
        std          14.896843
        min           0.000000
        25%         156.100000
        50%         167.405000
        75%         178.067000
        max         214.311000
        Name: 1_AIT_001_PV, dtype: float64
        '''

        local_df = self.normal_labels_df
        x_train = split_time_series_data(local_df['1_AIT_001_PV'], 50, 100)
        genetic_extractor = GeneticExtractor(population_size=50, iterations=25, verbose=True,
                                             mutation_prob=0.3, crossover_prob=0.3, max_shaps=9,
                                             wait=10, max_len=len(x_train) // 2)

        # Fit the GeneticExtractor and construct distance matrix
        # shapelets = genetic_extractor.fit(x_train, numpy.zeros(x_train.shape[0]))
        try:
            shapelets = genetic_extractor.fit(x_train[:50], numpy.zeros(50))
        except Exception as e :
            print(e)
        self.assertEqual(True, False)

    @ignore_warnings
    def test_with_multivariate(self):
        columns = ['1_AIT_001_PV', '1_AIT_002_PV']
        step = 500
        window_length = 1000
        step4negative = 10
        min_negative_last_chunk_size = 100
        x_train, x_test, y_train, y_test = self.__prepare_train_test_data(columns, step, window_length, step4negative, min_negative_last_chunk_size)
        for i in range(len(columns)):
            print(str(datetime.datetime.now()) + " start GENDIS for column '{0}'".format(columns[i]))
            x_c_train, x_c_test = x_train[:, i, :].T, x_test[:, i, :].T  # shape (x_n_ts, window_length)
            # Fit the GeneticExtractor and construct distance matrix
            genetic_extractor = GeneticExtractor(population_size=5, iterations=10, verbose=True, n_jobs=1,
                                                 mutation_prob=0.3, crossover_prob=0.3, normed=True,
                                                 wait=5, max_len=len(x_c_train) // 2)

            genetic_extractor.fit(x_c_train, y_train)
            print(str(datetime.datetime.now()) + ' shapelets ')
            print(str(datetime.datetime.now()) + ' shapelets # ' + str(len(genetic_extractor.shapelets)) + ' shape ' + str(genetic_extractor.shapelets[0].shape))
            distances_train = genetic_extractor.transform(x_c_train)
            print(str(datetime.datetime.now()) + ' distances_train')
            distances_test = genetic_extractor.transform(x_c_test)
            print(str(datetime.datetime.now()) + ' distances_test')

            # Fit ML classifier on constructed distance matrix
            lr = LogisticRegression()
            lr.fit(distances_train, y_train)
            print(str(datetime.datetime.now()) + ' lr.fit')
            print(str(datetime.datetime.now()) + ' Accuracy = {}'.format(accuracy_score(y_test, lr.predict(distances_test))))

    def test_random_shapelets(self):
        from tslearn.generators import random_walk_blobs
        from sklearn.linear_model import LogisticRegression
        np.random.seed(1337)
        X, y = random_walk_blobs(n_ts_per_blob=20, sz=64, noise_level=0.1)
        X = np.reshape(X, (X.shape[0], X.shape[1]))
        extractor = GeneticExtractor(iterations=5, population_size=10, location=True, n_jobs=1, verbose=True)
        distances = extractor.fit_transform(X, y)
        lr = LogisticRegression()
        _ = lr.fit(distances, y)
        lr.score(distances, y)
        self.assertEqual(1, lr.score(distances, y))

    def __prepare_train_test_data(self, columns: List[str], step: int = 500, time_window: int = 1000, step4negative: int = 5, min_negative_last_chunk_size: int = 20):
        negative_label_multivariate_split_data, neg_y = split_time_series_with_negative_labels(self.mixed_labels_df[columns],
                                                                                        self.mixed_labels_df['Attack LABLE (1:No Attack, -1:Attack)'],
                                                                                        step4negative, time_window, min_negative_last_chunk_size)

        # note we can't do split_time_series_with_labels() on combined DF of both dataset as they aren't chronological.
        mixed_label_multivariate_split_data, mixed_y = split_time_series_with_labels(self.mixed_labels_df[columns],
                                                                                     self.mixed_labels_df['Attack LABLE (1:No Attack, -1:Attack)'],
                                                                                     step, time_window)
        norm_label_multivariate_split_data, norm_y = split_time_series_with_labels(self.normal_labels_df[columns],
                                                                                   np.ones(self.normal_labels_df.size),
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

        # split into 2 equal portions
        if all_multivariate_prepared_x.shape[2] % 2 == 1:
            all_multivariate_prepared_x = all_multivariate_prepared_x[:, :, :-1]
            all_multivariate_y = all_multivariate_y[:-1]
        all_x_split_arr = np.dsplit(all_multivariate_prepared_x, 2)
        x_train = all_x_split_arr[0]
        x_test = all_x_split_arr[1]
        all_y_split_arr = np.hsplit(all_multivariate_y, 2)
        y_train = all_y_split_arr[0]
        y_test = all_y_split_arr[1]
        return x_train, x_test, y_train, y_test

    @ignore_warnings
    def test_with_multivariate_and_save_transformed(self):
        columns = ['1_AIT_001_PV', '1_AIT_002_PV']
        step = 500
        window_length = 1000
        step4negative = 5
        min_negative_last_chunk_size = 100
        x_train, y_train = self.__prepare_x_y_data(self.normal_labels_train_df, self.mixed_labels_train_df,
                                                   columns, step, window_length, step4negative, min_negative_last_chunk_size)
        x_test, y_test = self.__prepare_x_y_data(self.normal_labels_test_df, self.mixed_labels_test_df,
                                                 columns, step, window_length, step4negative, min_negative_last_chunk_size)
        for i in range(len(columns)):
            print(str(datetime.datetime.now()) + " start GENDIS for column '{0}'".format(columns[i]))
            x_c_train, x_c_test = x_train[:, i, :].T, x_test[:, i, :].T  # shape (x_n_ts, window_length)
            # Fit the GeneticExtractor and construct distance matrix
            genetic_extractor = GeneticExtractor(population_size=5, iterations=10, verbose=True, n_jobs=1,
                                                 mutation_prob=0.3, crossover_prob=0.3,
                                                 wait=5, max_len=len(x_c_train) // 2)

            genetic_extractor.fit(x_c_train, y_train)
            print(str(datetime.datetime.now()) + ' shapelets ')
            print(str(datetime.datetime.now()) + ' shapelets # ' + str(len(genetic_extractor.shapelets)) + ' shape ' + str(genetic_extractor.shapelets[0].shape))
            distances_train = genetic_extractor.transform(x_c_train)
            print(str(datetime.datetime.now()) + ' distances_train')
            distances_test = genetic_extractor.transform(x_c_test)
            print(str(datetime.datetime.now()) + ' distances_test')

            # Fit ML classifier on constructed distance matrix
            lr = LogisticRegression()
            lr.fit(distances_train, y_train)
            print(str(datetime.datetime.now()) + ' lr.fit')
            print(str(datetime.datetime.now()) + ' Accuracy = {}'.format(accuracy_score(y_test, lr.predict(distances_test))))

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

    def test_with_multivariate_and_save_transformed_2(self):

        conf = ShapeletsConfig(os.getcwd() + os.path.sep + "test_configuration_3")
        conf.step = 250
        conf.window_length = 500
        conf.min_negative_last_chunk_size = 50
        conf.population_size = 5
        conf.iterations = 10
        conf.wait = 5
        conf.normed = True
        conf.update()
        multi_var_shape_extractor = MultiVarShapeletsExtractor(conf, self.normal_labels_train_df,
                                                               self.mixed_labels_train_df,
                                                               self.normal_labels_test_df,
                                                               self.mixed_labels_test_df)

        multi_var_shape_extractor.prepare_data(list(self.mixed_labels_train_df.columns[3:18]))
        # multi_var_shape_extractor.prepare_data(['2_FIC_201_CO', '2_MCV_201_CO','2B_AIT_002_PV', '2A_AIT_001_PV','3_AIT_004_PV','2_PIC_003_PV'])
        multi_var_shape_extractor.discover_shapelets()

    def test_with_multivariate_train_test_classifier_with_transformed(self):

        conf = ShapeletsConfig(os.getcwd() + os.path.sep + "test_configuration_xxx")
        multi_var_shape_extractor = MultiVarShapeletsExtractor(conf, self.normal_labels_train_df,
                                                               self.mixed_labels_train_df,
                                                               self.normal_labels_test_df,
                                                               self.mixed_labels_test_df)

        # run train only after all the columns shapelets are done and saved test folder
        multi_var_shape_extractor.train_classifier()

    @classmethod
    def setUpClass(cls):
        cls.normal_labels_df = pd.read_csv("../Data/Cleaned_Trainset.csv")
        normal_df_mid_point = int(cls.normal_labels_df.shape[0] / 2)
        cls.mixed_labels_df = pd.read_csv("../Data/Cleaned_Testset.csv")
        mixed_df_mid_point = int(cls.mixed_labels_df.shape[0] / 2)

        cls.normal_labels_train_df = cls.normal_labels_df[:normal_df_mid_point]
        cls.normal_labels_test_df = cls.normal_labels_df[normal_df_mid_point + 1:]

        cls.mixed_labels_train_df = cls.mixed_labels_df[:mixed_df_mid_point]
        cls.mixed_labels_test_df = cls.mixed_labels_df[mixed_df_mid_point + 1:]

    @classmethod
    def tearDownClass(cls):
        pass


if __name__ == '__main__':
    unittest.main()
