import pandas as pd
import os
import pickle
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from dataclasses import dataclass


train = pd.read_csv(r'C:\Users\himay\OneDrive\Desktop\NeuroTraff\artifacts\train_data_transformed.csv')
test = pd.read_csv(r'C:\Users\himay\OneDrive\Desktop\NeuroTraff\artifacts\test_data_transformed.csv')
x_train = train.drop(columns=["delay ratio","Delay","Traffic level"])
x_test = test.drop(columns=["delay ratio", "Delay","Traffic level"])
y_train = train["Traffic level"]
y_test = test["Traffic level"]

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
class ModelTrainer:
    def train_predict(self,x_train, x_test, y_train, y_test, models, param_grids):
        best_model_obj = None
        best_model_name = None
        best_f1_score = 0
        f1_scores = {}
    
        for name, model in models.items():
            print(f"\nTraining {name}...")
            grid = GridSearchCV(model, param_grids[name], cv=5, n_jobs=-1)
            grid.fit(x_train, y_train)
        
            y_pred_test = grid.predict(x_test)
            test_f1 = f1_score(y_test, y_pred_test, average='weighted')
            f1_scores[name] = test_f1

            y_pred_train = grid.predict(x_train)
            train_f1 = f1_score(y_train, y_pred_train, average='weighted')
        
            print(f"Train weighted F1 score: {train_f1:.4f}")
            print(f"Test weighted F1 score: {test_f1:.4f}")
        
            if test_f1 > best_f1_score:
                best_f1_score = test_f1
                best_model_obj = grid.best_estimator_
                best_model_name = name
                print(f"Best Model Name: {best_model_name} with Test F1: {best_f1_score:.4f}")
                print(f"Best Estimator: {best_model_obj}")
                print(f"Best Parameters: {grid.best_params_}")

        # Save the best model to a pickle file
        if best_model_obj:
            model_path = os.path.join('artifacts', 'best_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(best_model_obj, f)
            print(f"\n✅ Best model saved to: {model_path}")

        return best_model_name, best_model_obj, f1_scores


trainer = ModelTrainer()
print(trainer.train_predict(x_train,x_test,y_train,y_test,models,param_grids))