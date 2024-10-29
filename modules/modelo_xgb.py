import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from fixing_data.preprocess import Preprocess
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd




class Xgb():

    def __init__(self, route: str):

        self.route = route
        self.df = Preprocess(self.route, "PERSONAL")
        self.personal = self.df.personal
        self.df_pre = self.df.preprocessed_data(self.personal)
        self.data = self.df.get_dummies(self.df_pre)
        self.X = self.data.drop(columns="cb_person_default_on_file")
        self.y = self.data["cb_person_default_on_file"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state=0, test_size=0.3)


    def model(self):
        model = XGBClassifier(
            objective='multi:softmax',  # For multi-class classification
            learning_rate=0.1,
            n_estimators=100,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False  # Disable label encoding warning for newer versions
        )

        model.fit(self.X_train, self.y_train)

        y_prob = model.predict_proba(self.X_test)[:, 1]
        y_pred = model.predict(self.X_test)

        return model, y_pred, y_prob


    def metrics(self):

        model, y_pred, y_prob = self.model()
        # Calcular las métricas
        accuracy = accuracy_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)

        metrics: list = [accuracy, recall, precision, f1]
        metrics_names: list = ["accuracy", "recall", "precision", "f1"]
        index: int = 0

        for metric in metrics:
            print(f"The score {metrics_names[index]} is: {metric}")
            index += 1


    def optimize_model(self):
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        param_grid = {
            'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
            'n_estimators': [50, 100, 200, 500, 1000],
            'max_depth': [3, 5, 7, 10, 15],
            'min_child_weight': [1, 3, 5, 10],
            'subsample': [0.5, 0.7, 0.8, 1],
            'colsample_bytree': [0.5, 0.7, 0.8, 1],
            'gamma': [0, 0.1, 0.5, 1, 5],
            'reg_lambda': [0, 0.1, 1, 10],
            'reg_alpha': [0, 0.1, 1, 10]
        }
        # Set up GridSearchCV
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='accuracy',
            cv=5,
            verbose=1,
            n_jobs=-1

        )

        grid_search.fit(self.X_train, self.y_train)

        print("Best Parameters:", grid_search.best_params_)
        print("Best Cross-Validation Score:", grid_search.best_score_)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        print("Test Accuracy:", accuracy)


if __name__ == "__main__":

    route = "./data/data.csv"
    log = Xgb(route)
    log.model()
    log.metrics()




