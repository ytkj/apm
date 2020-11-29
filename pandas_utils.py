from typing import Union, List

import numpy as np
import pandas as pd


DF = type(pd.DataFrame)
ND = type(np.ndarray)
S = type(pd.Series)


def missing_rate(df: pd.DataFrame) -> pd.Series:
    return df.isnull().sum() / len(df)


def is_pd(x: Union[DF, ND]) -> bool:
    return type(x) is pd.DataFrame


def pd_average(df_ys: List[pd.DataFrame]) -> pd.DataFrame:
    f = df_ys[0]
    return pd.DataFrame({
        f.columns[0]: f.iloc[:, 0],
        f.columns[1]: np.average([df_y.iloc[:, 1] for df_y in df_ys], axis=0)
    }, index=f.index)
