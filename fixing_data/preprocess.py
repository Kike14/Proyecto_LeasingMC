import numpy as np
import pandas as pd
from io import BytesIO
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

class Preprocess():

    def __init__(self, file: BytesIO, selected_column: str):

        self.selected_column = selected_column
        self.opened_file = self._load_file(file)
        self.personal = self._get_info()


    def _load_file(self, file_path: str) -> BytesIO:

        """
        Load the file based on its extension.
        :param file_path: Path to the file to be loaded
        :return: DataFrame if successfully loaded, otherwise None
        """
        if file_path.endswith(".xlsx"):
            try:
                return pd.read_excel(file_path, engine='openpyxl')
            except Exception as e:
                print(f"Error loading .xlsx file {file_path}: {e}")
        elif file_path.endswith(".xls"):
            try:
                return pd.read_excel(file_path, engine='xlrd')
            except Exception as e:
                print(f"Error loading .xls file {file_path}: {e}")
        elif file_path.endswith(".csv"):
            try:
                return pd.read_csv(file_path)
            except Exception as e:
                print(f"Error loading CSV file {file_path}: {e}")
        else:
            print(f"Unsupported file type for {file_path}. Skipping.")
            return None


    def _get_info(self) -> pd.DataFrame :

        df = self.opened_file
        df = df[df["loan_intent"] == self.selected_column]

        return df

    def null_proportion(self) -> pd.DataFrame:

        null_proportion = dict()

        for column in self.opened_file.columns:

            null_proportion[column] = round(
                len(self.opened_file[self.opened_file[column].isnull()]) / len(self.opened_file), 3)

        null_proportion["Total"] = round(self.opened_file.isnull().sum().sum() / self.opened_file.size, 3)
        null_df = pd.DataFrame(list(null_proportion.items()), columns=['Column', 'Null Proportion'])

        return null_df


    def change_outliers(self, dataframe: pd.DataFrame) -> pd.DataFrame:

        no_outliers_df = dataframe.copy()

        columns = ['person_age', 'person_emp_length']

        for column in columns:

            median = no_outliers_df[column].median()
            no_outliers_df.loc[:, column] = no_outliers_df.loc[:, column].apply(lambda x: median if x > 100 else x)


        return no_outliers_df

    def fill_null_values(self, dataframe: pd.DataFrame) -> pd.DataFrame:

        filled_df = dataframe.copy()

        for column in dataframe.columns:

            if dataframe.loc[:, column].dtype == 'object':
                continue

            else:
                median = dataframe.loc[:, column].median()
                dataframe.loc[:, column] = dataframe.loc[:, column].apply(lambda x: median if pd.isna(x) else x)

        return dataframe


    def distribution(self) -> Figure:

        numeric_columns = self.personal.select_dtypes(include="number")

        num_columns = 3
        num_rows = (len(numeric_columns.columns) + num_columns - 1) // num_columns  # Calculate the needed rows
        fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, num_rows * 5))  # Adjust figsize as needed
        axes = axes.flatten()  # Flatten the axes array for easy indexing

        for idx, column in enumerate(numeric_columns):
            sns.histplot(numeric_columns[column], kde=True, ax=axes[idx])
            axes[idx].set_title(f'Distribution of {column}')

        for i in range(len(numeric_columns.columns), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout(h_pad=5)
        plt.show()



    def distribution_with_changes(self, old_dataframe: pd.DataFrame, new_dataframe: pd.DataFrame) -> Figure:
        numeric_columns = old_dataframe.select_dtypes(include="number").columns
        num_columns = 3
        num_rows = (len(numeric_columns) + num_columns - 1) // num_columns

        fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, num_rows * 5))
        axes = axes.flatten()

        for idx, column in enumerate(numeric_columns):

            sns.kdeplot(old_dataframe[column], ax=axes[idx], color="blue", label="Original", fill=True)
            sns.kdeplot(new_dataframe[column], ax=axes[idx], color="red", label="Modified", fill=True)

            axes[idx].set_title(f'Distribution of {column}')
            axes[idx].legend()

        for i in range(len(numeric_columns), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout(h_pad=5)
        plt.show()
        return fig


    def preprocessed_data(self, dataframe: pd.DataFrame) -> pd.DataFrame:

        df = dataframe.copy()
        no_out_df = self.change_outliers(df)
        fill_nu_df = self.fill_null_values(no_out_df)


        return fill_nu_df



    def get_dummies(self, dataframe: pd.DataFrame):

        df = dataframe.copy()

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
            
        def cat_dti(percentage):
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

        df = df.drop(columns="loan_intent")
        df['person_age'] = df['person_age'].apply(categorias)
        df['person_income'] = df['person_income'].apply(cat_ingresos)
        df['loan_percent_income'] = df['loan_percent_income'].apply(cat_dti)
        # 1 cayo en impago, 0 no cayo en impago
        df['cb_person_default_on_file'] = (df['cb_person_default_on_file'] != 'N').astype(int)
        dummy_columns: list = ['person_age', "person_income", 'person_home_ownership', 'loan_grade', 'loan_percent_income']
        df = pd.get_dummies(df, columns=dummy_columns)
        for col in df.select_dtypes(include='bool'):
            df[col] = df[col].astype(int)

        return df




pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 20)


if __name__ == "__main__":
    route: str = "../data/data.csv"
    data = Preprocess(route, "PERSONAL")
    old = data.personal
    new = data.preprocessed_data(old)
    print(old)
    print(new)
    print(data.distribution_with_changes(old_dataframe=old, new_dataframe=new))
