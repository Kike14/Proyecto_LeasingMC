import numpy as np
import pandas as pd
from io import BytesIO
import seaborn as sns
import matplotlib.pyplot as plt
from fixing_data.preprocess import Preprocess

path = "C:/Users/Usuario/OneDrive/Escritorio/credit_risk_dataset.csv"

if __name__ == "__main__":
    route: str = path
    data= Preprocess(path, "PERSONAL")
    old= data.personal
    df = data.preprocessed_data(old)
    # print(old)
    # print(new)
    # print(data.distribution_with_changes(old,new))

def categorias(edad):
    if edad < 18:
        return 'menor_de_edad'
    elif 18 <= edad <= 25:
        return 'jovenes'
    elif 26 <= edad <= 35:
        return 'adulto_joven'
    elif 36 <= edad <= 45:
        return 'adulto'
    elif 46 <= edad <= 55:
        return 'adulto_mayor'
    else:
        return 'tercera_edad'

# Función corregida para clasificar ingresos
def cat_ingresos(income):
    if income <= 5000:
        return 'low_income'
    elif 5001 <= income <= 10000:
        return "medium_low_income"
    elif 10001 <= income <= 20000:
        return "medium_income"
    elif 20001 <= income <= 50000:
        return "medium_high_income"
    elif 50001 <= income <= 100000:
        return "high_income"
    else:
        return "super_high_income"

df['age_categories'] = df['person_age'].apply(categorias)
df['income_categories']= df['person_income'].apply(cat_ingresos)

df_dummies_age = pd.get_dummies(df['age_categories'], prefix='age')
df_dummies_income = pd.get_dummies(df['income_categories'], prefix='income')
df_dummies_hos = pd.get_dummies(df['person_home_ownership'], prefix= 'hos')


df_dummies = pd.concat([df, df_dummies_age], axis=1)
df_dummies = pd.concat([df_dummies, df_dummies_income], axis=1)
df_dummies = pd.concat([df_dummies, df_dummies_hos], axis=1)
df_dummies.drop(['person_age', 'age_categories', 'income_categories', 'person_home_ownership'], axis=1, inplace=True)


print(df_dummies.columns)

fig, axes = plt.subplots(2, 2, figsize=(12, 6))  # 1 fila, 2 columnas

# Gráfica 1: Histograma con KDE
sns.histplot(df['age_categories'], ax=axes[0][0], color='skyblue')
sns.histplot(df['person_age'], ax=axes[0][1])
sns.histplot(df['income_categories'], ax=axes[1][0])
sns.histplot(df['person_income'], ax=axes[1][1])


# Ajustar la separación entre las subplots
plt.tight_layout()

# Mostrar las gráficas
plt.show()




