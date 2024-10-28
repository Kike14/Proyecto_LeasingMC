import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from fixing_data.preprocess import Preprocess
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix


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

        # Hacer predicciones
        y_pred = model.predict(self.X_test)

        return model, y_pred  # Devuelve tanto el modelo como las predicciones

    def metrics(self):
        # Obtener modelo y predicciones
        model, y_pred = self.model()

        # Calcular las métricas
        accuracy = accuracy_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)

        metrics = [accuracy, recall, precision, f1]
        metrics_names = ["accuracy", "recall", "precision", "f1"]

        for name, metric in zip(metrics_names, metrics):
            print(f"The score {name} is: {metric}")


if __name__ == "__main__":

    route = "./data/data.csv"
    log = Lg(route)
    log.model()
    log.metrics()



