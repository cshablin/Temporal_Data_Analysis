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
        # ax.set(ylabel="# shapelets", xlabel='sliding window size', title='Number of discovered shapelets during training')
        plt.xticks(fontsize= 16)
        plt.yticks(fontsize= 16)
        plt.suptitle('Number of discovered shapelets during training', fontsize=18)
        plt.xlabel('sliding window size', fontsize=17)
        plt.ylabel('# shapelets', fontsize=17)
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

        # ax.set(ylabel="shapelet length", xlabel='sliding window size', title='Discovered shapelets lengths during training')
        plt.xticks(fontsize= 16)
        plt.yticks(fontsize= 16)
        plt.suptitle('Discovered shapelets lengths during training', fontsize=18)
        plt.xlabel('sliding window size', fontsize=17)
        plt.ylabel('shapelet length', fontsize=17)

        # Get the handles and labels. For this example it'll be 2 tuples
        # of length 4 each.
        handles, labels = ax.get_legend_handles_labels()

        # When creating the legend, only use the first two elements
        # to effectively remove the last two.
        # plt.legend(title='Smoker', loc='upper left', labels=['Hell Yeh', 'Nah Bruh'])
        plt.legend(handles[0:2], labels[0:2], loc='upper left',
                   title='Minimum trailing\nsuccessive attack\nin positive samples') # labels=['25 sec', '50 sec'], labelcolor=['green', 'red']
        plt.show()

    def test_plot_feature_importance(self):
        xgb_path = 'test_configuration_6_win_100//XGBClassifier_0.64.joblib'
        from joblib import load
        xgb = load(xgb_path)
        importance = xgb.feature_importances_
        # summarize feature importance
        # for i,v in enumerate(importance):
        #     print('Feature: %0d, Score: %.5f' % (i,v))
        # plot feature importance
        # plt.bar([x for x in range(len(importance))], importance, log=True)
        fig, ax = plt.subplots()
        ax.bar([x for x in range(len(importance))], importance, log=True, ec="k",)# align="edge"
        # ax.set(ylabel="coefficient", xlabel='Feature', title='Localized Shapelets Feature importance in XGBoost')
        plt.xticks(fontsize= 20)
        plt.yticks(fontsize= 20)
        fig.suptitle('Localized Shapelets Feature importance in XGBoost', fontsize=25)
        plt.xlabel('Feature', fontsize=23)
        plt.ylabel('coefficient', fontsize=23)
        plt.show()

    def test_feature_importance_prepare_reduced(self):
        xgb_path = 'test_configuration_6_win_100//XGBClassifier_0.64.joblib'
        from joblib import load
        xgb = load(xgb_path)
        importance = xgb.feature_importances_


        shapelet_df = pd.read_csv('num_of_shapelets.csv') # (labor_df['State and area']=='Hawaii') | (labor_df['State and area']=='Maryland')
        shapelet_df = shapelet_df[shapelet_df['trailing_negatives']==50]
        shapelet_df = shapelet_df[shapelet_df['window']==100]

        week_columns = []
        col_index = 0
        last_coef_index = 0
        for col_num_of_shapes in shapelet_df['num_of_shapelets']:
            col_important = False
            for i in range(col_num_of_shapes * 2):  # distance + location
                coef = importance[last_coef_index + i]
                if coef > 0.0001:
                    col_important = True
                    break
            if not col_important:
                week_columns.append(col_index)
            last_coef_index += col_num_of_shapes * 2
            col_index += 1

        columns_names = np.load('test_configuration_6_win_100' + os.path.sep + 'columns.npy')
        week_columns_names = np.take(columns_names, week_columns, axis=0)
        strong_columns_names = set(list(columns_names)) - set(list(week_columns_names))
        np.save('filtered_columns.npy', np.array(list(strong_columns_names)))


if __name__ == '__main__':
    unittest.main()
