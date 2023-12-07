import numpy as np

MODEL_HYPERPARAMETERS_DEF = \
    {
        "Logistic Regression": \
            {
                "penalty": "l2",
                "C": 1.,
                "solver": "liblinear",
                "multi_class": "auto",
                "n_jobs": None,
                "max_iter": 100
            },
        "SVM":
            {
                "kernel": "linear",
                "C": 1.,
                "degree": 3
            },
        "DecisionTreeClassifier":
            {
                "criterion": "gini",
                "splitter": "best",
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "min_weight_fraction_leaf": 0.,
                "max_features": None,
                "random_state": None,
                "max_leaf_nodes": None,
                "min_impurity_decrease": 0.,
                "class_weight": None,
                "ccp_alpha": 0.
            },
        "RandomForestClassifier":{
            "n_estimators":100,
            "criterion":"gini",
            "max_depth":None,
            "min_samples_split":2,
            "min_samples_leaf":1,
            "min_weight_fraction_leaf":0.0,
            "max_features":"sqrt",
            "max_leaf_nodes":None,
            "min_impurity_decrease":0.0,
            "bootstrap":True,
            "oob_score":False,
            "n_jobs":None,
            "random_state":None,
            "verbose":0,
            "warm_start":False,
            "class_weight":None,
            "ccp_alpha":0.0,
            "max_samples":None

        },
        "GradientBoostingClassifier":
            {
                "loss":"log_loss",
                "learning_rate":0.1,
                "n_estimators":100,
                "subsample":1.0,
                "criterion":"friedman_mse",
                "min_samples_split":2,
                "min_samples_leaf":1,
                "min_weight_fraction_leaf":0.0,
                "max_depth":3,
                "min_impurity_decrease":0.0,
                "init":None,
                "random_state":None,
                "max_features":None,
                "verbose":0,
                "max_leaf_nodes":None,
                "warm_start":False,
                "validation_fraction":0.1,
                "n_iter_no_change":None,
                "tol":1e-4,
                "ccp_alpha":0.0
            },
        "XGBClassifier":
            {
                'num_class':2,
                'max_depth':5,
                'n_estimators':10,
                'learning_rate':0.1,
                'objective':'multi:softmax',
                'subsample':0.8,
            },
        "LinearDiscriminantAnalysis":
            {
                "solver":'lsqr',
                "shrinkage":0
            },
        "QuadraticDiscriminantAnalysis":
            {
                'reg_param': [0.5],
            }

    }

PARAMS_GRID = \
    {
        "Logistic Regression": \
            {
                "penalty": ["l1", "l2"],
                "C": np.logspace(-3, 3, 7),
                "solver": ["liblinear"],
                "multi_class": ["auto"],
                "n_jobs": [None],
                "max_iter": [500, 700, 1000]
            },
        "SVM":
            {
                "kernel": ["poly", "rbf", "sigmoid"],
                "C": [0.1, 1, 10, 100],
                "degree": [3],
                "gamma": [1, 0.1, 0.01, 0.001]

            },
        "DecisionTreeClassifier":
            {
                "criterion": ["gini"],
                "splitter": ["best"],
                "max_depth": [None],
                "min_samples_split": [2, 3, 4],
                "min_samples_leaf": [1],
                "min_weight_fraction_leaf": [0.],
                "max_features": [None],
                "random_state": [None],
                "max_leaf_nodes": [None]+list(range(2, 100)),
                "min_impurity_decrease": [0.],
                "class_weight": [None],
                "ccp_alpha": [0.]
            },
        "RandomForestClassifier":
            {
                "n_estimators":list(np.random.randint(50,500,size=5)),
                "max_depth":list(np.random.randint(1,20,size=5))
            },
        "GradientBoostingClassifier":
            {
                "learning_rate":[0.15,0.1,0.05,0.01,0.005,0.001],
                "n_estimators":[100,250,500,750,1000,1250,1500,1750]
            },
        "XGBClassifier":
            {
                "learning_rate":[0.1, 0.01],
                "n_estimators":[100,250,500],
                "max_depth":list(np.random.randint(1,20,size=5)),
                "gamma":[0.01,0],
                "min_child_weight":[6,8],
                "reg_alpha":[0.1,0.01]
            },
        "LinearDiscriminantAnalysis":
            {
                "solver":['lsqr', 'eigen'],
                "shrinkage":np.arange(0, 1, 0.01)
            },
        "QuadraticDiscriminantAnalysis":
            {
                'reg_param': [0.1, 0.2, 0.3, 0.4, 0.5],
            }
    }

