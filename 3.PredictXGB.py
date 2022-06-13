import json
from collections import Counter

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, GridSearchCV, train_test_split

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from Models.Metrics import scoring_list
from Models.ParameterGrids import RF_parameter_grid, XGB_parameter_grid
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

import xgboost as xgb

from pylab import rcParams
import numpy as np
np.seterr(divide='ignore')

from numpy.random import seed
from sklearn.ensemble import RandomForestClassifier

seed(7)

import wandb
wandb.init(project="NoROCPython")


rcParams['figure.figsize'] = 8, 6
LABELS = ["0", "1"]

def main () :
    configs = json.load(open('Utils/Configuration.json', 'r'))
    data_path = configs['paths']['data_path']
    timeseries_path = data_path+"Training/"
    output_path = data_path+"Output/"

    features = configs['data']['features']
    outcome = configs['data']['classification_outcome']

    prevalence_rates = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    for p in prevalence_rates:
        output_file = output_path+"XGB_weighted"+str(p)+".json"
        data = pd.read_csv(timeseries_path+"TimeSeries"+str(p)+".csv")
        X = data[features]
        y = data[outcome].astype('int')

        xgb_grid = XGB_parameter_grid()
        clf = xgb.XGBClassifier(eval_metric='error')
        grid_search = GridSearchCV(estimator=clf, param_grid=xgb_grid,
                                   cv=3, n_jobs=-1, verbose=2)

        X_traintest, X_tune, y_traintest, y_tune = train_test_split(X, y, test_size = 0.33, random_state = 42)
        imputer = IterativeImputer(max_iter=20, random_state=0)
        scaler = MinMaxScaler()
        X_imputed = imputer.fit_transform(X_tune)
        X_scaled = scaler.fit_transform(X_imputed)
        grid_search.fit(X_scaled, y_tune)
        best_model = grid_search.best_estimator_

        steps = list()
        steps.append(('scaler', MinMaxScaler()))
        steps.append(('imputer', IterativeImputer(max_iter=20, random_state=0)))
        steps.append(('model', clf))
        pipeline = Pipeline(steps=steps)

        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        # evaluate the model using cross-validation
        cv_scores = cross_validate(pipeline, X_traintest, y_traintest, scoring=scoring_list, cv=cv, n_jobs=-1)
        cv_scores = {k : v.tolist() for k, v in cv_scores.items()}
        out_file = open(output_file, "w")
        json.dump(cv_scores, out_file, indent=6)
if __name__ == '__main__' :
    main()
