import pandas as pd
import os
import pickle
import sys
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import f1_score
from dataclasses import dataclass
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.exception import CustomException

class ModelTrainer:
    try:
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
                    data_dir = os.path.join('artifacts', 'Piplines')
                    model_path = os.path.join(data_dir, 'best_model.pkl')
                    with open(model_path, 'wb') as f:
                        pickle.dump(best_model_obj, f)
                    print(f"\n✅ Best model saved to: {model_path}")

            return best_model_name, best_model_obj, f1_scores
    except Exception as e:
            raise CustomException(sys,e)


