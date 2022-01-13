import os
import unittest
import warnings
import numpy as np
import pandas as pd

from typing import List
from Common.data_preparation import split_time_series_with_labels, split_time_series_with_negative_labels
from Exp_Shapelets.Config import ShapeletsConfig
from Exp_Shapelets.classifier_helper import get_logit_pipe_grid, get_nn_pipe_grid, get_forest_pipe_grid
from Exp_Shapelets.discovery_of_shapelets import MultiVarShapeletsExtractor


def ignore_warnings(test_func):
    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test_func(self, *args, **kwargs)
    return do_test


class ClassifierTestCase1(unittest.TestCase):

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

    def __write_report(self, file_name, txt):
        with open(file_name, "w") as text_file:
            text_file.write(txt)

    def __get_shape_extractor(self, conf_test):
        conf = ShapeletsConfig(os.getcwd() + os.path.sep + conf_test)
        multi_var_shape_extractor = MultiVarShapeletsExtractor(conf, self.normal_labels_train_df,
                                                               self.mixed_labels_train_df,
                                                               self.normal_labels_test_df,
                                                               self.mixed_labels_test_df)
        return multi_var_shape_extractor

    def __run_forest(self, conf_test):
        multi_var_shape_extractor = self.__get_shape_extractor(conf_test)
        report = multi_var_shape_extractor.train_test_classifier_grid_search(get_forest_pipe_grid())
        self.__write_report('TestReports' + os.path.sep + 'FR_' + conf_test + '.txt', report)

    def __run_nn(self, conf_test):
        multi_var_shape_extractor = self.__get_shape_extractor(conf_test)
        report = multi_var_shape_extractor.train_test_classifier_grid_search(get_nn_pipe_grid())
        self.__write_report('TestReports' + os.path.sep + 'NN_' + conf_test + '.txt', report)

    def __run_lg(self, conf_test):
        multi_var_shape_extractor = self.__get_shape_extractor(conf_test)
        report = multi_var_shape_extractor.train_test_classifier_grid_search(get_logit_pipe_grid())
        self.__write_report('TestReports' + os.path.sep + 'LR_' + conf_test + '.txt', report)

    @ignore_warnings
    def test_configuration_3_win_500(self):
        conf_test = "test_configuration_3_win_500"
        # self.__run_lg(conf_test)
        self.__run_nn(conf_test)
        self.__run_forest(conf_test)

    @ignore_warnings
    def test_configuration_3_win_500_25(self):
        conf_test = "test_configuration_3_win_500_25"
        # self.__run_lg(conf_test)
        self.__run_nn(conf_test)
        self.__run_forest(conf_test)

    @ignore_warnings
    def test_configuration_4_win_250(self):
        conf_test = "test_configuration_4_win_250"
        # self.__run_lg(conf_test)
        self.__run_nn(conf_test)
        self.__run_forest(conf_test)

    @ignore_warnings
    def test_configuration_4_win_250_25(self):
        conf_test = "test_configuration_4_win_250_25"
        # self.__run_lg(conf_test)
        self.__run_nn(conf_test)
        self.__run_forest(conf_test)

    @ignore_warnings
    def test_configuration_5_win_750(self):
        conf_test = "test_configuration_5_win_750"
        # self.__run_lg(conf_test)
        self.__run_nn(conf_test)
        self.__run_forest(conf_test)

    @ignore_warnings
    def test_configuration_5_win_750_25(self):
        conf_test = "test_configuration_5_win_750_25"
        # self.__run_lg(conf_test)
        self.__run_nn(conf_test)
        self.__run_forest(conf_test)

    @ignore_warnings
    def test_configuration_6_win_100(self):
        conf_test = "test_configuration_6_win_100"
        # self.__run_lg(conf_test)
        self.__run_nn(conf_test)
        self.__run_forest(conf_test)

    @ignore_warnings
    def test_configuration_6_win_100_25(self):
        conf_test = "test_configuration_6_win_100_25"
        # self.__run_lg(conf_test)
        self.__run_nn(conf_test)
        self.__run_forest(conf_test)

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
