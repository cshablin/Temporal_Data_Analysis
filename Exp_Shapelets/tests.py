import unittest
import numpy
import numpy as np
import pandas as pd
from gendis.genetic import GeneticExtractor
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from typing import List
from Common.data_preparation import split_time_series_data, split_time_series_with_labels, \
    split_time_series_with_negative_labels


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

    def test_with_multivariate(self):
        columns = ['1_AIT_001_PV', '1_AIT_002_PV']
        step = 500
        window_length = 1000
        step4negative = 50
        min_negative_last_chunk_size = 5
        x_train, x_test, y_train, y_test = self.__prepare_train_test_data(columns, step, window_length, step4negative, min_negative_last_chunk_size)
        for i in range(len(columns)):
            print("start GENDIS for column '{0}'".format(columns[i]))
            x_c_train, x_c_test = x_train[:, i, :].T, x_test[:, i, :].T  # shape (x_n_ts, window_length)
            # Fit the GeneticExtractor and construct distance matrix
            genetic_extractor = GeneticExtractor(population_size=5, iterations=10, verbose=True, n_jobs=8,
                                                 mutation_prob=0.3, crossover_prob=0.3,
                                                 wait=5, max_len=len(x_c_train) // 2)
            shapelets = genetic_extractor.fit(x_c_train, y_train)
            distances_train = genetic_extractor.transform(x_c_train)
            distances_test = genetic_extractor.transform(x_c_test)

            # Fit ML classifier on constructed distance matrix
            lr = LogisticRegression()
            lr.fit(distances_train, y_train)
            print('Accuracy = {}'.format(accuracy_score(y_test, lr.predict(distances_test))))

    def test_random_shapelets(self):
        from tslearn.generators import random_walk_blobs
        from sklearn.linear_model import LogisticRegression
        np.random.seed(1337)
        X, y = random_walk_blobs(n_ts_per_blob=20, sz=64, noise_level=0.1)
        X = np.reshape(X, (X.shape[0], X.shape[1]))
        extractor = GeneticExtractor(iterations=5, population_size=10, location=True)
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
        all_x_split_arr = np.dsplit(all_multivariate_prepared_x, 2)
        x_train = all_x_split_arr[0]
        x_test = all_x_split_arr[1]
        all_y_split_arr = np.hsplit(all_multivariate_y, 2)
        y_train = all_y_split_arr[0]
        y_test = all_y_split_arr[1]
        return x_train, x_test, y_train, y_test

    @classmethod
    def setUpClass(cls):
        cls.normal_labels_df = pd.read_csv("../Data/Cleaned_Trainset.csv")
        cls.mixed_labels_df = pd.read_csv("../Data/Cleaned_Testset.csv")

    @classmethod
    def tearDownClass(cls):
        pass


if __name__ == '__main__':
    unittest.main()
