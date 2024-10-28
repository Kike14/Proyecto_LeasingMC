from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from Proyecto_LeasingMC.fixing_data.preprocess import Preprocess
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

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state=0, test_size=0.3)


    def model(self):
        model = XGBClassifier(
            objective='binary:logistic',  # For binary classification; use 'multi:softmax' for multi-class
            n_estimators=100,  # Number of trees
            learning_rate=0.1,  # Step size shrinkage
            max_depth=6,  # Maximum tree depth for base learners
            random_state=42  # Seed for reproducibility
        )
        model.fit(self.X_train, self.y_train)

        # Hacer predicciones
        y_pred = model.predict(self.X_test)

        return y_pred


    def metrics(self):

        y_pred = self.model()
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



if __name__ == "__main__":

    route = "../data/data.csv"
    log = Lg(route)
    log.model()
    log.metrics()




