from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from Proyecto_LeasingMC.fixing_data.preprocess import Preprocess
import pandas as pd

route: str = "../data/data.csv"
df = Preprocess(route, "PERSONAL")
personal = df.personal
df_pre = df.preprocessed_data(personal)
data = df.get_dummies(df_pre)


if __name__ == "__main__":
    print(data.columns)
    print(data)
