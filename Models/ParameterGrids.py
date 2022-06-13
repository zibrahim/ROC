import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

def SVM_parameter_grid ():
    space={
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': [3],
        'class_weight': ['balanced']
    }
    return space

max_depth = [int(x) for x in np.linspace(10, 110, num=20)]
#gamma = [int(x) for x in np.linspace(1,9)]
#reg_alpha = [int(x) for x in np.linspace(40,180)]
#reg_lambda = [x for x in np.linspace(0,1, num=0.05)]
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=400)]

def XGB_parameter_grid ():
    space = {'max_depth' : max_depth,
             #'gamma' : gamma,
             #'reg_alpha' : reg_alpha,
             #'reg_lambda' : reg_lambda,
             #'colsample_bytree' : hp.uniform('colsample_bytree', 0.5, 1),
             #'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
             #'n_estimators' :n_estimators
             #'seed' : 7,
             }
    return space

def RF_parameter_grid () :
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=400)]
    # Number of features to consider at every split
    max_features = ['auto']
    # Maximum number of levels in tree
    #max_depth = [int(x) for x in np.linspace(10, 110, num=40)]
    #max_depth.append(None)
    # Minimum number of samples required to split a node
    #min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [10, 50, 100]
    # Create the random grid
    random_grid = {'max_features' : max_features,
                   'min_samples_leaf' : min_samples_leaf,
                   'class_weight': ['balanced']
                   }
    return random_grid
