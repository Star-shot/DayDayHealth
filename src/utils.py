import pandas as pd
import numpy as np

def load_data(filepath):
    """
    此函数用于加载数据集，支持 xlsx 和 csv 文件，会删除有缺失值的数据，默认最后一列是标签。

    参数:
    filepath (str): 数据集文件的路径，支持 xlsx 和 csv 格式。

    返回:
    tuple: 包含特征矩阵 X 和标签向量 y 的元组。
    """
    if filepath.endswith('.xlsx'):
        df = pd.read_excel(filepath)
    elif filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        raise ValueError("不支持的文件格式，仅支持 .xlsx 和 .csv 文件。")
    
    # 删除包含缺失值的行
    df = df.dropna()
    
    # 提取特征和标签
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    return X, y