# import sys
# import os
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(project_root)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from fixing_data.preprocess import Preprocess
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import numpy as np


class Lg():

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
        y_pred = model.predict(self.X_test)

        return model, y_pred, y_prob

    def metrics(self):

        model, y_pred, y_prob = self.model()

        # Define a range of thresholds to test
        thresholds = [0.5]

        # Store the results
        results = []

        for threshold in thresholds:

            y_pred = (y_prob >= threshold).astype(int)

            accuracy = accuracy_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)

            # Store results for each threshold
            results.append((threshold, accuracy, precision, recall, f1))

        results_df = pd.DataFrame(results, columns=["Threshold", "Accuracy", "Precision", "Recall", "F1-Score"])

        return results_df
        # metrics = [accuracy, recall, precision, f1]
        # metrics_names = ["accuracy", "recall", "precision", "f1"]
        #
        # for name, metric in zip(metrics_names, metrics):
        #     print(f"The score {name} is: {metric}")

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 30)


if __name__ == "__main__":

    route = "../data/data.csv"
    log = Lg(route)
    log.model()
    metric = log.metrics()
    print(log)
    print(metric)



