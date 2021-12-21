import unittest
import numpy
import numpy as np
import pandas as pd
from gendis.genetic import GeneticExtractor
from sklearn.model_selection import train_test_split

from Common.data_preparation import split_time_series_data, split_time_series_with_labels


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
        # self.normal_labels_df['Attack LABLE (1:No Attack, -1:Attack)'] = np.ones(self.normal_labels_df.size)
        norm_label_multivariate_split_data, norm_y = split_time_series_with_labels(self.normal_labels_df['1_AIT_001_PV', '1_AIT_002_PV'],
                                                                           np.ones(self.normal_labels_df.size))
        mixed_label_multivariate_split_data, mixed_y = split_time_series_with_labels(self.mixed_labels_df['1_AIT_001_PV', '1_AIT_002_PV'],
                                                                            self.mixed_labels_df['Attack LABLE (1:No Attack, -1:Attack)'])
        # to stack row wise, the multivariate(df.columns) dimension is the third
        all_multivariate_prepared_x = np.stack((norm_label_multivariate_split_data, mixed_label_multivariate_split_data), axis=0)
        all_multivariate_y = np.append(norm_y, mixed_y)
        x_train, x_test, y_train, y_test = train_test_split(all_multivariate_prepared_x, all_multivariate_y, test_size=0.30, random_state=12)

    def test_random_shapelets(self):
        from tslearn.generators import random_walk_blobs
        from sklearn.linear_model import LogisticRegression
        import numpy as np
        np.random.seed(1337)
        X, y = random_walk_blobs(n_ts_per_blob=20, sz=64, noise_level=0.1)
        X = np.reshape(X, (X.shape[0], X.shape[1]))
        extractor = GeneticExtractor(iterations=5, population_size=10, location=True)
        distances = extractor.fit_transform(X, y)
        lr = LogisticRegression()
        _ = lr.fit(distances, y)
        lr.score(distances, y)
        self.assertEqual(1, lr.score(distances, y))

    @classmethod
    def setUpClass(cls):
        cls.normal_labels_df = pd.read_csv("../Data/Cleaned_Trainset.csv")
        cls.mixed_labels_df = pd.read_csv("../Data/Cleaned_Testset.csv")

    @classmethod
    def tearDownClass(cls):
        pass


if __name__ == '__main__':
    unittest.main()
