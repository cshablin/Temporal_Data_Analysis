import unittest
import pandas as pd
import numpy as np
from gendis.genetic import GeneticExtractor


class ShapeletsTestCase(unittest.TestCase):

    def test_shapelets_in_col(self):
        #load training dataset
        local_df = self.train_df
        # Univariate time series example
        # Creating a GeneticExtractor object
        # genetic_extractor = GeneticExtractor(population_size=50, iterations=25, verbose=True,
        #                                      mutation_prob=0.3, crossover_prob=0.3,
        #                                      wait=10, max_len=len(X_train) // 2)
        #
        # # Fit the GeneticExtractor and construct distance matrix
        # shapelets = genetic_extractor.fit(X_train, y_train)
        self.assertEqual(True, False)

    @classmethod
    def setUpClass(cls):
        cls.train_df = pd.read_csv("../Data/WADI_14days_new.csv")

    @classmethod
    def tearDownClass(cls):
        pass


if __name__ == '__main__':
    unittest.main()
