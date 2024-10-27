import numpy as np
import pandas as pd
from io import BytesIO


class Preprocess():

    def __init__(self, file: BytesIO):

        self.file = file
        self.opened_file = self.load_file(file)


    def load_file(self, file_path: str) -> BytesIO:

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


    def get_info(self) -> pd.DataFrame :

        df = self.opened_file
        df = df[df["loan_intent"] == "PERSONAL"]

        return df



pd.set_option('display.width', 100)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 20)

if __name__ == "__main__":

    route: str = "data/data.csv"
    data = Preprocess("data/data.csv")
    personal = data.get_info()
    print(personal)

