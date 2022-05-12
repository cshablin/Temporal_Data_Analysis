from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, make_scorer, recall_score
import numpy as np
from sklearn.svm import SVC
from xgboost import XGBClassifier

random_state = 10

# f1 = make_scorer(f1_score, average='macro')
f1 = make_scorer(f1_score, average='binary')
# f1 = make_scorer(recall_score, average='binary')


def get_forest_pipe_grid() -> GridSearchCV:
    forest_pipe = Pipeline([
        ('classifier' , RandomForestClassifier(random_state=random_state)),
    ])

    param_grid ={
        'classifier__n_estimators' : list(range(50, 101, 10)),
        'classifier__max_features' : [0.15, 0.2, 0.25], # 0.05, 0.1,
        'classifier__max_depth' : list(range(5, 8, 1)),
    }

    return GridSearchCV(forest_pipe, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1, scoring=f1)


def get_logit_pipe_grid():
    logistic_pipe = Pipeline([
        ('classifier' , LogisticRegression(random_state=random_state)),
    ])

    param_grid ={
        'classifier__penalty' : ['l1', 'l2'], # , 'elasticnet'
        'classifier__C' : np.logspace(-4, 4, 10)[2:9],
        'classifier__solver' : ['lbfgs', 'liblinear'], # , 'saga'
        # 'classifier__l1_ratio' : [None, 0.1],
    }

    return GridSearchCV(logistic_pipe, param_grid = param_grid, cv=5, verbose=True, n_jobs=-1, scoring=f1)


def get_nn_pipe_grid():
    pipe = Pipeline([
        ('classifier', MLPClassifier(solver='adam',max_iter = 100, early_stopping=True,verbose=0, random_state=random_state))
    ])
    # par = {'classifier__alpha': [0.0005], 'classifier__batch_size': [128], 'classifier__early_stopping': [True], 'classifier__hidden_layer_sizes': [(64, 32, 16, 8)], 'classifier__learning_rate': ['adaptive'], 'classifier__solver': ['adam']}
    param_grid = {
        'classifier__hidden_layer_sizes': [(64,32,16,8)], # (64,),(8,8) BEST (64,32,16,8)
        'classifier__alpha': [0.0001, 0.0005, 0.002],  # np.logspace(-4, -2, 4)
        'classifier__solver': ['adam'], # ,'lbfgs', 'sgd'
        'classifier__batch_size': [128],
        'classifier__early_stopping': [True], #  False
        'classifier__learning_rate': ['adaptive'], # 'invscaling', 'constant','adaptive'
    }
    return GridSearchCV(pipe, param_grid, cv=5, verbose=0, n_jobs=-1, scoring=f1)


def get_svm_pipe_grid():
    tuned_parameters = [
        {"kernel": ["rbf"], "gamma": [  0.001, 0.0001], "C": [ 1000], 'max_iter': [1000]}, # , 'auto'
        # {"kernel": ["linear"], "C": [0.1, 1, 10, 100, 1000], 'max_iter': [1000]},
        # {"kernel": ["poly"], "C": [0.1, 1, 10, 100, 1000], 'degree': [2, 3], "gamma": [1, 0.1, 0.01, 0.001, 0.0001], 'max_iter': [1000]}, # 10, 100, 1000
        # {"kernel": ["poly"], "C": [10, 100], 'degree': [2], "gamma": ['scale'], 'max_iter': [100]}, # 10, 100, 1000
        # {"kernel": ["poly"], "C": [100, 1000], 'degree': [3,4], "gamma": ['scale']}, # 10, 100, 1000
    ]
    return GridSearchCV(SVC(probability=True, random_state=random_state), tuned_parameters, cv=None, verbose=0, n_jobs=-1, scoring=f1)
    # return RandomizedSearchCV(SVC(probability=True,random_state=random_state), tuned_parameters, random_state=random_state, cv=5, verbose=0, n_jobs=-1, scoring=f1)


def get_gb_pipe_grid():
    tuned_parameters = [
        {
            "loss":["deviance"],
            "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
            "min_samples_split": np.linspace(0.1, 0.5, 12),
            "min_samples_leaf": np.linspace(0.1, 0.5, 12),
            "max_depth":[3,5,8],
            "max_features":["log2","sqrt"],
            "criterion": ["friedman_mse",  "mae"],
            "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
            "n_estimators":[11]
        },
        # {
        #     "loss":["deviance"],
        #     "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
        #     "min_samples_split": np.linspace(0.1, 0.5, 12),
        #     "min_samples_leaf": np.linspace(0.1, 0.5, 12),
        #     "max_depth":[3,5,8],
        #     "max_features":["log2","sqrt"],
        #     "criterion": ["friedman_mse",  "mae"],
        #     "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
        #     "n_estimators":[50]
        # },
    ]
    return RandomizedSearchCV(GradientBoostingClassifier(random_state=random_state), tuned_parameters, random_state=random_state, cv=5, verbose=0, n_jobs=-1, scoring=f1)
    # return GridSearchCV(GradientBoostingClassifier(random_state=random_state), tuned_parameters, cv=None, verbose=0, n_jobs=-1, scoring=f1)


def get_xgb_pipe_grid():
    tuned_parameters = {
        # 'learning_rate': [0.01, 0.02, 0.005],
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [4, 5, 6]
    }
    tuned_parameters = {'subsample': [0.9, 1.0], 'min_child_weight': [0.9, 1, 1.1], 'max_depth': [5,6], 'gamma': [0.8, 1, 1.2], 'colsample_bytree': [0.5, 0.6, 0.7]}
    # tuned_parameters = {'subsample': [0.9, 1.0, 1.1], 'min_child_weight': [4, 5, 6], 'max_depth': [5, 6], 'gamma': [1.5], 'colsample_bytree': [0.8]}
    folds = 3
    param_comb = 108  # 20
    np.random.seed(random_state)
    xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic', silent=True, nthread=1, random_state=random_state)
    return RandomizedSearchCV(xgb, param_distributions=tuned_parameters, n_iter=param_comb, scoring=f1, n_jobs=-1, cv=folds, verbose=3, random_state=random_state )


def get_knn_pipe_grid():
    tuned_parameters = [
        {"n_neighbors": [7, 9, 15],  # 7, 9, 15
         "weights": ['uniform', 'distance'], # 'uniform',
         'metric': ['minkowski']}, # 10, 100, 100
    ]
    return RandomizedSearchCV(KNeighborsClassifier(), tuned_parameters, random_state=random_state, cv=None, verbose=0, n_jobs=-1, scoring=f1)


def get_ada_boost_pipe_grid():
    tuned_parameters = [
        {
         #    "base_estimator__criterion" : ["gini", "entropy"],
         # "base_estimator__splitter" :   ["best", "random"],
         # 'base_estimator__max_depth':[i for i in range(2, 11, 2)],
         # 'base_estimator__min_samples_leaf':[5, 10],
         "n_estimators": [100],
         "algorithm": ['SAMME'], # , 'SAMME.R'
         'learning_rate':[0.01, 0.1, 0.5],
         }

    ]
    # return RandomizedSearchCV(AdaBoostClassifier(random_state=random_state), tuned_parameters, random_state=random_state, cv=None, verbose=0, n_jobs=-1, scoring=f1)
    return GridSearchCV(AdaBoostClassifier(random_state=random_state), tuned_parameters, cv=None, verbose=0, n_jobs=-1, scoring=f1)
