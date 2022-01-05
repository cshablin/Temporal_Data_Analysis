from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np


def get_forest_pipe_grid() -> GridSearchCV:
    forest_pipe = Pipeline([
        ('classifier' , RandomForestClassifier()),
    ])

    param_grid ={
        'classifier__n_estimators' : list(range(50,101,10)),
        'classifier__max_features' : [0.05, 0.1, 'auto'],
        'classifier__max_depth' : list(range(5, 7, 1)),
    }

    return GridSearchCV(forest_pipe, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)


def get_logit_pipe_grid():
    logistic_pipe = Pipeline([
        ('classifier' , LogisticRegression()),
    ])

    param_grid ={
        'classifier__penalty' : ['l1', 'l2'], # , 'elasticnet'
        'classifier__C' : np.logspace(-4, 4, 10)[0:7],
        'classifier__solver' : ['lbfgs', 'liblinear'], # , 'saga'
        # 'classifier__l1_ratio' : [None, 0.1],
    }

    return GridSearchCV(logistic_pipe, param_grid = param_grid, cv=5, verbose=True, n_jobs=-1)