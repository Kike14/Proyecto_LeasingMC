from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from Proyecto_LeasingMC.fixing_data.preprocess import Preprocess
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV


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
        model = RandomForestClassifier()
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

if __name__ == "__main__":
    route = "../data/data.csv"
    forest = Forest(route)
    forest.optimize_model()
    # forest.model()
    # forest.metrics()



