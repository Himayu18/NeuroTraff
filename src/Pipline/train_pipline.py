import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.components.model_trainer import ModelTrainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
models = {
        'RandomForestClassifier': RandomForestClassifier(random_state=42),
        'AdaBoostClassifier': AdaBoostClassifier(random_state=42),
        'GradientBoostingClassifier': GradientBoostingClassifier(random_state=42),
        'XGBClassifier': XGBClassifier(random_state=42),
        'CatBoostClassifier': CatBoostClassifier(random_state=42, verbose=0)
    }
param_grids = {
        'RandomForestClassifier': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'bootstrap': [True]
        },
        'AdaBoostClassifier': {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1]
        },
        'GradientBoostingClassifier': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        },
        'XGBClassifier': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        },
        'CatBoostClassifier': {
            'iterations': [100, 200],
            'learning_rate': [0.01, 0.1],
            'depth': [4, 6],
            'l2_leaf_reg': [1, 3]
        }
    }
train = pd.read_csv(r'artifacts\TransformedData\train_data_transformed.csv')
test = pd.read_csv(r'artifacts\TransformedData\test_data_transformed.csv')
x_train = train.drop(columns=["delay ratio","Delay","Traffic level"])
x_test = test.drop(columns=["delay ratio", "Delay","Traffic level"])
y_train = train["Traffic level"]
y_test = test["Traffic level"]

trainer = ModelTrainer()
print(trainer.train_predict(x_train,x_test,y_train,y_test,models,param_grids))