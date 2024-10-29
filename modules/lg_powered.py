# import sys
# import os
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(project_root)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from fixing_data.preprocess import Preprocess
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd


class Lg_powered():

    def __init__(self, route: str):
        self.route = route
        self.df = Preprocess(self.route, "PERSONAL")
        self.personal = self.df.personal
        self.df_pre = self.df.preprocessed_data(self.personal)
        self.data = self.df.get_dummies(self.df_pre)
        self.X = self.data.drop(columns="cb_person_default_on_file")
        self.y = self.data["cb_person_default_on_file"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, random_state=0, test_size=0.3
        )

    def model(self):
        # Entrenar el modelo
        model = LogisticRegression(max_iter=200)  # Aumentamos max_iter para evitar advertencias
        model.fit(self.X_train, self.y_train)

        y_prob = model.predict_proba(self.X_test)[:, 1]

        # Hacer predicciones
        y_pred = model.predict(self.X_test)

        return model, y_pred, y_prob

    def metrics(self):

        model, y_pred, y_prob = self.model()

        # Define a range of thresholds to test
        thresholds = np.arange(0.0, 1.05, 0.05)

        # Store the results
        results = []

        for threshold in thresholds:
            # Apply threshold to get predictions
            y_pred = (y_prob >= threshold).astype(int)

            accuracy = accuracy_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)

            # Store results for each threshold
            results.append((threshold, accuracy, precision, recall, f1))

        results_df = pd.DataFrame(results, columns=["Threshold", "Accuracy", "Precision", "Recall", "F1-Score"])

        return results_df

    def optimize_model(self):
        model = LogisticRegression()
        param_grid = {
            'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'solver': ['liblinear', 'saga'],
            'class_weight': [None, 'balanced'],
            'max_iter': [100, 200, 500, 1000],
            'l1_ratio': [0.1, 0.5, 0.7, 0.9, 1]
        }

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


pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 30)


if __name__ == "__main__":

    route = "../data/data.csv"
    log = Lg_powered(route)
    # log.model()
    log.optimize_model()
    # metric = log.metrics()
    # print(log)
    # print(metric)

# Best Parameters: {'C': 0.01, 'class_weight': 'balanced', 'l1_ratio': 0.1, 'max_iter': 100, 'penalty': 'l2', 'solver': 'newton-cg'}
# Best Cross-Validation Score: 0.8343668098854474
# Test Accuracy: 0.8346409173204586
