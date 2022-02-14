import os
import unittest
from typing import List, Any, Tuple
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
        result_lengths = pd.DataFrame(columns=['shapelet_length', 'window', 'trailing_negatives'])
        for folder, conf in conf_tests.items():
            rows, lengths_concat = self.__get_shapelets_analysis(folder)
            df_lengths = pd.DataFrame(lengths_concat, columns=['shapelet_length'])
            df_lengths['window'] = conf[0]
            df_lengths['trailing_negatives'] = conf[1]
            result_lengths = result_lengths.append(df_lengths)

            df = pd.DataFrame(rows, columns=columns[0:3])
            df['window'] = conf[0]
            df['trailing_negatives'] = conf[1]
            result = result.append(df)
        result_lengths.to_csv('shapelets_lengths.csv', index=False)
        result.to_csv('num_of_shapelets.csv', index=False)

    def __get_shapelets_analysis(self, conf_folder: str) -> Tuple[List[List[Any]], List[Any]]:
        rows = []
        columns_names = np.load(conf_folder + os.path.sep + 'columns.npy')
        col_index = 0
        lengths_concat = []
        for col in columns_names:
            row = [col_index]
            col_model_path = conf_folder + os.path.sep + col + os.path.sep + 'model.p'
            genetic_extractor = GeneticExtractor.load(col_model_path)
            shapelets = genetic_extractor.shapelets
            row.append(len(shapelets))
            lengths = []
            for shapelet in shapelets:
                lengths.append(shapelet.shape[0])
            lengths_concat.extend(lengths)
            row.append(lengths)
            rows.append(row)
            col_index += 1
        return rows, lengths_concat

    def test_plot_num_of_shaplets(self):
        df_results = pd.read_csv("num_of_shapelets.csv")
        columns = ['column_index', 'num_of_shapelets', 'shapelets_lengths_array', 'window', 'trailing_negatives']
        g_1 = df_results.groupby(['column_index', 'window', 'trailing_negatives']).agg({'num_of_shapelets': 'mean'})
        g_1 = g_1.reset_index()
        palette = sns.color_palette("tab10", 4)
        # ax = sns.lineplot(y="num_of_shapelets", x="column_index", hue="window", style='trailing_negatives', palette=palette,  data=g_1)
        ax = sns.boxplot(y="num_of_shapelets", x="window", hue="trailing_negatives", palette=palette,  data=df_results, showmeans= True,meanprops={"marker":"o",
                                                                                                                                            "markerfacecolor":"white",
                                                                                                                                            "markeredgecolor":"black",
                                                                                                                                            "markersize":"5"})
        ax.set(ylabel="# shapelets", xlabel='sliding window size', title='Number of discovered shapelets during training')
        plt.legend(loc='upper left', title='Minimum trailing\nsuccessive attack\nin positive samples')
        plt.show()

    def test_plot_shaplets_lengths_distribution(self):
        df = pd.read_csv("shapelets_lengths.csv")
        columns=['shapelet_length', 'window', 'trailing_negatives']
        palette = sns.color_palette("tab10", 4)
        # ax = sns.catplot(x="window", y="shapelet_length", hue="trailing_negatives",
        #             kind="violin", inner="stick", split=True,
        #             palette=palette, data=df)

        # ax2 = sns.stripplot(x="window", y="shapelet_length", hue="trailing_negatives",
        #                     color = 'black', dodge=True,
        #                     alpha = 0.05,
        #                     data = df)

        ax = sns.boxplot(x="window", y="shapelet_length", hue="trailing_negatives", palette=palette,  data=df, showmeans= True,meanprops={"marker":"o",
                                                                                                                                          "markerfacecolor":"white",
                                                                                                                                          "markeredgecolor":"black",
                                                                                                                                          "markersize":"5"})

        # ax = sns.catplot(x="window", y="shapelet_length", hue="trailing_negatives",
        #             kind="violin", bw=.15, cut=0, split=True,
        #             palette=palette, data=df)
        # ax = sns.catplot(x="window", y="shapelet_length", hue="trailing_negatives",
        #                  palette=palette, data=df)
        # ax = sns.catplot(x="day", y="total_bill", hue="sex", kind="swarm", data=

        ax.set(ylabel="shapelet length", xlabel='sliding window size', title='Discovered shapelets lengths during training')

        # Get the handles and labels. For this example it'll be 2 tuples
        # of length 4 each.
        handles, labels = ax.get_legend_handles_labels()

        # When creating the legend, only use the first two elements
        # to effectively remove the last two.
        # plt.legend(title='Smoker', loc='upper left', labels=['Hell Yeh', 'Nah Bruh'])
        plt.legend(handles[0:2], labels[0:2], loc='upper left',
                   title='Minimum trailing\nsuccessive attack\nin positive samples') # labels=['25 sec', '50 sec'], labelcolor=['green', 'red']
        plt.show()


if __name__ == '__main__':
    unittest.main()
