import os
import unittest
import warnings
import pandas as pd

from Exp_Shapelets.Config import ShapeletsConfig
from Exp_Shapelets.classifier_helper import get_logit_pipe_grid, get_nn_pipe_grid, get_forest_pipe_grid, \
    get_svm_pipe_grid, get_gb_pipe_grid, get_knn_pipe_grid, get_ada_boost_pipe_grid
from Exp_Shapelets.discovery_of_shapelets import MultiVarShapeletsExtractor


def ignore_warnings(test_func):
    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test_func(self, *args, **kwargs)
    return do_test


class ClassifierTestCase(unittest.TestCase):

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

    def __run_svm(self, conf_test):
        multi_var_shape_extractor = self.__get_shape_extractor(conf_test)
        report = multi_var_shape_extractor.train_test_classifier_grid_search(get_svm_pipe_grid())
        self.__write_report('TestReports' + os.path.sep + 'SVM_' + conf_test + '.txt', report)

    def __run_gb(self, conf_test):
        multi_var_shape_extractor = self.__get_shape_extractor(conf_test)
        report = multi_var_shape_extractor.train_test_classifier_grid_search(get_gb_pipe_grid())
        self.__write_report('TestReports' + os.path.sep + 'GB_' + conf_test + '.txt', report)

    def __run_ada_b(self, conf_test):
        multi_var_shape_extractor = self.__get_shape_extractor(conf_test)
        report = multi_var_shape_extractor.train_test_classifier_grid_search(get_ada_boost_pipe_grid())
        self.__write_report('TestReports' + os.path.sep + 'ADA_B_' + conf_test + '.txt', report)

    def __run_knn(self, conf_test):
        multi_var_shape_extractor = self.__get_shape_extractor(conf_test)
        report = multi_var_shape_extractor.train_test_classifier_grid_search(get_knn_pipe_grid())
        self.__write_report('TestReports' + os.path.sep + 'KNN_' + conf_test + '.txt', report)

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
        # self.__run_svm(conf_test)
        # self.__run_ada_b(conf_test)
        # self.__run_gb(conf_test)
        # self.__run_knn(conf_test)
        # self.__run_forest(conf_test)

    @ignore_warnings
    def test_configuration_3_win_500_25(self):
        conf_test = "test_configuration_3_win_500_25"
        # self.__run_lg(conf_test)
        self.__run_nn(conf_test)
        # self.__run_svm(conf_test)
        # self.__run_ada_b(conf_test)
        # self.__run_gb(conf_test)
        # self.__run_knn(conf_test)
        # self.__run_forest(conf_test)


    @ignore_warnings
    def test_configuration_4_win_250(self):
        conf_test = "test_configuration_4_win_250"
        # self.__run_lg(conf_test)
        self.__run_nn(conf_test)
        # self.__run_svm(conf_test)
        # self.__run_ada_b(conf_test)
        # self.__run_gb(conf_test)
        # self.__run_knn(conf_test)
        # self.__run_forest(conf_test)


    @ignore_warnings
    def test_configuration_4_win_250_25(self):
        conf_test = "test_configuration_4_win_250_25"
        # self.__run_lg(conf_test)
        self.__run_nn(conf_test)
        # self.__run_svm(conf_test)
        # self.__run_ada_b(conf_test)
        # self.__run_gb(conf_test)
        # self.__run_knn(conf_test)
        # self.__run_forest(conf_test)


    @ignore_warnings
    def test_configuration_5_win_750(self):
        conf_test = "test_configuration_5_win_750"
        # self.__run_lg(conf_test)
        self.__run_nn(conf_test)
        # self.__run_svm(conf_test)
        # self.__run_ada_b(conf_test)
        # self.__run_gb(conf_test)
        # self.__run_knn(conf_test)
        # self.__run_forest(conf_test)


    @ignore_warnings
    def test_configuration_5_win_750_25(self):
        conf_test = "test_configuration_5_win_750_25"
        # self.__run_lg(conf_test)
        self.__run_nn(conf_test)
        # self.__run_svm(conf_test)
        # self.__run_ada_b(conf_test)
        # self.__run_gb(conf_test)
        # self.__run_knn(conf_test)
        # self.__run_forest(conf_test)


    @ignore_warnings
    def test_configuration_6_win_100(self):
        conf_test = "test_configuration_6_win_100"
        # self.__run_lg(conf_test)
        self.__run_nn(conf_test)
        # self.__run_svm(conf_test)
        # self.__run_ada_b(conf_test)
        # self.__run_gb(conf_test)
        # self.__run_knn(conf_test)
        # self.__run_forest(conf_test)


    @ignore_warnings
    def test_configuration_6_win_100_25(self):
        conf_test = "test_configuration_6_win_100_25"
        # self.__run_lg(conf_test)
        self.__run_nn(conf_test)
        # self.__run_svm(conf_test)
        # self.__run_ada_b(conf_test)
        # self.__run_gb(conf_test)
        # self.__run_knn(conf_test)
        # self.__run_forest(conf_test)


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
