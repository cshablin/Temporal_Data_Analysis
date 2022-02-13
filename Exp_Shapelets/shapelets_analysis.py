import os
import unittest
from typing import List, Any
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from gendis.genetic import GeneticExtractor


class ShapeletsAnalysisTest(unittest.TestCase):

    def test_prepare_shapelets_lengths_scv(self):
        conf_tests = {
            "test_configuration_6_win_100": (100, 50), "test_configuration_6_win_100_25": (100, 25),
            "test_configuration_4_win_250": (250, 50), "test_configuration_4_win_250_25": (250, 25),
            "test_configuration_3_win_500": (500, 50), "test_configuration_3_win_500_25": (500, 25),
            "test_configuration_5_win_750": (750, 50), "test_configuration_5_win_750_25": (750, 25),

        }
        columns = ['column_index', 'num_of_shapelets', 'shapelets_lengths_array', 'window', 'trailing_negatives']
        result = pd.DataFrame(columns=columns)
        for folder, conf in conf_tests.items():
            rows = self.__get_shapelets_analysis(folder)
            df = pd.DataFrame(rows, columns=columns[0:3])
            df['window'] = conf[0]
            df['trailing_negatives'] = conf[1]
            result = result.append(df)
        result.to_csv('shapelets_lengths.csv', index=False)

    def __get_shapelets_analysis(self, conf_folder: str) -> List[List[Any]]:
        rows = []
        columns_names = np.load(conf_folder + os.path.sep + 'columns.npy')
        col_index = 0
        for col in columns_names:
            row = [col_index]
            col_model_path = conf_folder + os.path.sep + col + os.path.sep + 'model.p'
            genetic_extractor = GeneticExtractor.load(col_model_path)
            shapelets = genetic_extractor.shapelets
            row.append(len(shapelets))
            lengths = []
            for shapelet in shapelets:
                lengths.append(shapelet.shape[0])
            row.append(lengths)
            rows.append(row)
            col_index += 1
        return rows

    def test_plot_num_of_shaplets(self):
        df_results = pd.read_csv("shapelets_lengths.csv")
        columns = ['column_index', 'num_of_shapelets', 'shapelets_lengths_array', 'window', 'trailing_negatives']
        g_1 = df_results.groupby(['column_index', 'window', 'trailing_negatives']).agg({'num_of_shapelets': 'mean'})
        g_1 = g_1.reset_index()
        palette = sns.color_palette("tab10", 4)
        # ax = sns.lineplot(y="num_of_shapelets", x="column_index", hue="window", style='trailing_negatives', palette=palette,  data=g_1)
        ax = sns.boxplot(y="num_of_shapelets", x="window", hue="trailing_negatives", palette=palette,  data=g_1)
        ax.set(ylabel="# shapelets", xlabel='sliding window size', title='number of discovered shapelets within all variables')
        plt.show()

    def test_plot_shaplets_lengths_distribution(self):
        sns.catplot(x="day", y="total_bill", hue="sex",
                    kind="violin", inner="stick", split=True,
                    palette="pastel", data=tips)


if __name__ == '__main__':
    unittest.main()
