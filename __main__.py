import numpy as np
import pandas as pd
from io import BytesIO
import seaborn as sns
import matplotlib.pyplot as plt


class Preprocess():

    def __init__(self, file: BytesIO):

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
        df = df[df["loan_intent"] == "PERSONAL"]

        return df

    def null_proportion(self) -> pd.DataFrame:

        null_proportion = dict()

        for column in self.opened_file.columns:

            null_proportion[column] = round(
                len(self.opened_file[self.opened_file[column].isnull()]) / len(self.opened_file), 3)

        null_proportion["Total"] = round(self.opened_file.isnull().sum().sum() / self.opened_file.size, 3)
        null_df = pd.DataFrame(list(null_proportion.items()), columns=['Column', 'Null Proportion'])

        return null_df

    def distribution(self):

        numeric_columns = self.personal.select_dtypes(include="number")
        for column in numeric_columns:
            sns.displot(numeric_columns[column], kde=True)
            plt.title(f'Distribution of {column}')
            plt.show()





pd.set_option('display.width', 100)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 20)

if __name__ == "__main__":

    route: str = "data/data.csv"
    data = Preprocess("data/data.csv")
    personal = data.personal
    # print(type(personal))
    print(data.distribution())


