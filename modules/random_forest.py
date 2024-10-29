# import sys
# import os
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(project_root)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from       fixing_data.preprocess import Preprocess
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd


class Forest():

    def __init__(self, route: str):
        self.route = route
        self.df = Preprocess(self.route, "PERSONAL")
        self.personal = self.df.personal
        self.df_pre = self.df.preprocessed_data(self.personal)
        self.data = self.df.get_dummies(self.df_pre)
        self.X = self.data.drop(columns="cb_person_default_on_file")
        self.y = self.data["cb_person_default_on_file"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state=0,
                                                                                test_size=0.3)

    def model(self):
        model = RandomForestClassifier(bootstrap=False,
                                       max_depth=10,
                                       max_features=None,
                                       min_samples_leaf =2,
                                       min_samples_split= 5,
                                       n_estimators= 200

                                       )

        model.fit(self.X_train, self.y_train)

        y_prob = model.predict_proba(self.X_test)[:, 1]

        # Define your custom threshold
        threshold = 0.5

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

        model = RandomForestClassifier()
        param_grid = {
            'max_features': ['sqrt', 'log2', None],  # Remove 'auto' and replace with valid options
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
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
        # Fit GridSearchCV
        grid_search.fit(self.X_train, self.y_train)

        # Retrieve best parameters and score
        print("Best Parameters:", grid_search.best_params_)
        print("Best Cross-Validation Score:", grid_search.best_score_)

        # Predict with the best model
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(self.X_test)

        # Evaluate
        accuracy = accuracy_score(self.y_test, y_pred)
        print("Test Accuracy:", accuracy)

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 30)


if __name__ == "__main__":
    route = "../data/data.csv"
    forest = Forest(route)
    # forest.optimize_model()
    forest.model()
    metric = forest.metrics()
    print(metric)

# Fitting 5 folds for each of 432 candidates, totalling 2160 fits
# Best Parameters: {'bootstrap': False, 'max_depth': 10, 'max_features': None, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 200}
# Best Cross-Validation Score: 0.8369541320070516
# Test Accuracy: 0.8352444176222088

