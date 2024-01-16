import os
import pandas as pd


def data_normalization(dataframe: pd.DataFrame):
    """
    use Min-Max normalization to normalize data

    :param dataframe:
    :return:
    """
    column_name_list = dataframe.columns[1:]
    for column in column_name_list:
        min_col = dataframe[column].min()
        max_col = dataframe[column].max()
        dataframe[column] = (dataframe[column] - min_col) / (max_col - min_col)


def add_lag_features(dataframe: pd.DataFrame):
    """
    add lag feature to the last column

    :param dataframe:
    :return:
    """
    dataframe["OT_lag_1"] = dataframe["OT"].shift(periods=1, fill_value=0)


def process_df(dataframe: pd.DataFrame, target_col: str = "OT"):
    """
    process the original data

    :params dataframe: pandas dataframe type
    :params target_col: prediction column OT(Oil Temparature)'
    :return:
    """
    data_normalization(dataframe)
    add_lag_features(dataframe)


if __name__ == '__main__':
    data_path = "data"
    for file_name in os.listdir(data_path):
        input_path = os.path.join(data_path, file_name)
        output_path = os.path.join(data_path, f"processed_{file_name}")
        data = pd.read_csv(input_path)
        process_df(data)
        data.to_csv(output_path, index = False)