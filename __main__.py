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


    def change_outliers(self) -> pd.DataFrame:

        columns = ['person_age', 'person_emp_length']

        for column in columns:

            median = self.personal[column].median()
            self.personal.loc[:, column] = self.personal.loc[:, column].apply(lambda x: median if x > 100 else x)


        return self.opened_file

    def fill_null_values(self) -> pd.DataFrame:

        for column in self.personal.columns:

            if self.personal.loc[:, column].dtype == 'object':
                continue

            else:

                median = self.personal.loc[:, column].median()

                self.personal.loc[:, column] = self.personal.loc[:, column].apply(lambda x: median if pd.isna(x) else x)


        return self.opened_file


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



pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 20)


if __name__ == "__main__":

    route: str = "data/data.csv"
    data = Preprocess("data/data.csv", "PERSONAL")
    old = data.personal
    data.change_outliers()
    new = data.fill_null_values()
    print(old)
    print(new)
    print(data.distribution_with_changes(old_dataframe=old, new_dataframe=new))
