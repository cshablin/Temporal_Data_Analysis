import unittest
import numpy
import pandas as pd
from gendis.genetic import GeneticExtractor

from Common.data_preparation import split_time_series_data


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

        local_df = self.train_df
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
        cls.train_df = pd.read_csv("../Data/Cleaned_Trainset.csv")

    @classmethod
    def tearDownClass(cls):
        pass


if __name__ == '__main__':
    unittest.main()
