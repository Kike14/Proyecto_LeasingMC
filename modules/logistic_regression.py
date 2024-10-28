from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from Proyecto_LeasingMC.fixing_data.preprocess import Preprocess
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

route: str = "../data/data.csv"
df = Preprocess(route, "PERSONAL")
personal = df.personal
df_pre = df.preprocessed_data(personal)
data = df.get_dummies(df_pre)

X = data.drop(columns="cb_person_default_on_file")
y = data["cb_person_default_on_file"]

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0, test_size=0.3)

model = LogisticRegression(max_iter=100)
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Calcular las métricas
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


if __name__ == "__main__":

    metrics: list = [accuracy, recall, precision, f1]
    metrics_names: list = ["accuracy", "recall", "precision", "f1"]
    index: int = 0

    for metric in metrics:
        print(f"The score {metrics_names[index]} is: {metric}")
        index += 1

