import numpy as np
import pandas as pd
from io import BytesIO
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("C:/Users/Usuario/OneDrive/Escritorio/credit_risk_dataset.csv")

mediana_age = df.loc[df['person_age'] <= 100, 'person_age'].median()
df['person_age'] = df['person_age'].apply(lambda x: mediana_age if x > 100 else x)

def categorias(edad):
    if edad < 18:
        return 'menor_de_edad'
    elif edad >= 18 < 25:
        return 'jovenes'
    elif edad >= 25 < 35:
        return 'adulto_joven'
    elif edad >= 35 < 50:
        return 'adulto'
    elif edad >= 50 < 65:
        return 'adulto_mayor'
    else: return 'tercera_edad'

df['categories'] = df['person_age'].apply(categorias)

df_dummies = pd.get_dummies(df['categories'], prefix='age')

df_dummies = pd.concat([df, df_dummies], axis=1)
df_dummies.drop(['person_age', 'categories'], axis=1, inplace=True)



print(df_dummies.head())

print(df)