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


# Clasificación de rangos de edad
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

# Clasificación de ingresos
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
    
    # Clasificacipon dti   
def categorizar_loan_percent(percentage):
    if percentage <= .20:
        return 'bajo_endeudamiento'
    elif .21 <= percentage <= .35:
        return 'moderado_endeudamiento'
    elif .36 <= percentage <= .50:
        return 'alto_endeudamiento'
    elif .51 <= percentage <= .70:
        return 'endeudamiento_critico'
    else:
        return 'sobreendeudado'

df['age_categories'] = df['person_age'].apply(categorias)
df['income_categories']= df['person_income'].apply(cat_ingresos)
df['debt_category'] = df['loan_percent_income'].apply(categorizar_loan_percent)

def generar_dummies(dataframe, columns):
    """Genera variables dummy para las columnas seleccionadas."""
    dummies_list = [pd.get_dummies(dataframe[col], prefix=col) for col in columns]
    return pd.concat([dataframe] + dummies_list, axis=1)


df_dummies = generar_dummies(df, ['age_categories', 'income_categories', 'person_home_ownership', 'debt_category'])


columnas_a_eliminar = [
        'person_age', 'age_categories', 'income_categories',
        'person_home_ownership', 'debt_category', 
        'loan_percent_income', 'person_income'
    ]
df_dummies.drop(columnas_a_eliminar, axis=1, inplace=True)

print(df_dummies.columns)





