import datetime
import os
import re
import sys
from collections import defaultdict
from gc import collect
sys.path.append("../util")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from feature import Feature, get_arguments, generate_features
Feature.dir = "../processed"

"""
前準備
rolling_feat.py
"""

from collections import deque
from math import sqrt
LAG = 28

def rolling_cor(X, Y, window=28, th=10):
    
    def var(sum_, sum2, N):
        return max(sum2/N - (sum_/N)**2, 0)
    
    if type(X) == pd.Series:
        X = X.values
    if type(Y) == pd.Series:
        Y = Y.values
    
    QX = deque()
    QY = deque()
    res = []
    X_sum = 0
    X_sum2 = 0
    Y_sum = 0
    Y_sum2 = 0
    XY_sum = 0
    XY_sum2 = 0
    N = 0
    start = 0
    for i in range(len(X)):
        x, y = X[i], Y[i]
        if pd.isna(x) or pd.isna(y):
            res.append(np.NaN)
            continue
        N += 1
        X_sum += x
        X_sum2 += x*x
        Y_sum += y
        Y_sum2 += y*y
        XY_sum += x+y
        XY_sum2 += (x+y)*(x+y)
        QX.append(x)
        QY.append(y)
        if (len(QX) > window):
            x, y = QX.popleft(), QY.popleft()
            X_sum -= x
            X_sum2 -= x*x
            Y_sum -= y
            Y_sum2 -= y*y
            XY_sum -= x+y
            XY_sum2 -= (x+y)*(x+y)
            N -= 1
        if (len(QX) >= th): 
            VX = var(X_sum, X_sum2, N)
            VY = var(Y_sum, Y_sum2, N)
            covXY = (var(XY_sum, XY_sum2, N) - VX - VY)/2
            res.append(covXY / (sqrt(VX*VY)+1e-9))
        else:
            res.append(np.NaN)
    return res

class Rolling_cor(Feature):
    def create_features(self):
        datas = []
        feat = pd.DataFrame()

        autocor_data = defaultdict(lambda : [])
        for id_, df in tqdm(data_df.groupby("id"), total=data_df["id"].nunique()):
            X = df[f"id_lag_{LAG}_rmean_1"]
            X_r7 = df[f"id_lag_{LAG}_rmean_7"]
            X_r28 = df[f"id_lag_{LAG}_rmean_28"]

            X_1y_r28 = df[f"id_lag_{LAG}_rmean_28"].shift(364)

            autocor_r1_d7_w91 = rolling_cor(X, X.shift(7), 91)
            autocor_r7_d28_w91 = rolling_cor(X_r7, X_r7.shift(28), 91)
            autocor_r28_d364_w91 = rolling_cor(X_r28, X_r28.shift(364), 91)

            item_id_r28_w91 = rolling_cor(X_r28, df[f"item_id_lag_{LAG}_rmean_28"])
            id_dept_r28_w91 = rolling_cor(X_r28, df[f"dept_id_lag_{LAG}_rmean_28"])
            id_store_r28_w91 = rolling_cor(X_r28, df[f"store_id_lag_{LAG}_rmean_28"])
            id_store_dept_r28_w91 = rolling_cor(X_r28, df[f"store_id_dept_id_lag_{LAG}_rmean_28"])

            for i, d in enumerate(df["d"].values):
                autocor_data["id"].append(id_)
                autocor_data["d"].append(d)
                autocor_data["id_autocor_r1_d7_w91"].append(autocor_r1_d7_w91[i])
                autocor_data["id_autocor_r7_d28_w91"].append(autocor_r7_d28_w91[i])
                autocor_data["id_autocor_r28_d364_w91"].append(autocor_r28_d364_w91[i])

                autocor_data["id_item_r28_w91"].append(item_id_r28_w91[i])
                autocor_data["id_dept_r28_w91"].append(id_dept_r28_w91[i])
                autocor_data["id_store_r28_w91"].append(id_store_r28_w91[i])
                autocor_data["id_store_dept_r28_w91"].append(id_store_dept_r28_w91[i])

        autocor_data = pd.DataFrame(dict(autocor_data))
        datas.append(data_df[["id", "d"]].merge(autocor_data,
                                                on=["id", "d"],
                                                how="left")\
                                         .drop(["id", "d"], axis=1)\
                                         .astype(np.float32))    
        del autocor_data
        collect()
        
        autocor_data = defaultdict(lambda : [])
        for item_id_, df in tqdm(data_df.groupby("item_id"), total=data_df["item_id"].nunique()):
            df = df.drop_duplicates(["item_id", "d"])
            X = df[f"id_lag_{LAG}_rmean_1"]
            X_r7 = df[f"id_lag_{LAG}_rmean_7"]
            X_r28 = df[f"id_lag_{LAG}_rmean_28"]

            X_1y_r28 = df[f"id_lag_{LAG}_rmean_28"].shift(364)

            autocor_r1_d7_w91 = rolling_cor(X, X.shift(7), 91)
            autocor_r7_d28_w91 = rolling_cor(X_r7, X_r7.shift(28), 91)
            autocor_r28_d364_w91 = rolling_cor(X_r28, X_r28.shift(364), 91)

            id_dept_r28_w91 = rolling_cor(X_r28, df[f"dept_id_lag_{LAG}_rmean_28"])

            for i, d in enumerate(df["d"].values):
                autocor_data["item_id"].append(item_id_)
                autocor_data["d"].append(d)
                autocor_data["item_id_autocor_r1_d7_w91"].append(autocor_r1_d7_w91[i])
                autocor_data["item_id_autocor_r7_d28_w91"].append(autocor_r7_d28_w91[i])
                autocor_data["item_id_autocor_r28_d364_w91"].append(autocor_r28_d364_w91[i])
                autocor_data["item_id_dept_r28_w91"].append(id_dept_r28_w91[i])

        autocor_data = pd.DataFrame(dict(autocor_data))
        datas.append(data_df[["item_id", "d"]].merge(autocor_data,
                                                on=["item_id", "d"],
                                                how="left")\
                                         .drop(["item_id", "d"], axis=1)\
                                         .astype(np.float32))
        

        del autocor_data
        collect()
        self.data = pd.concat(datas, axis=1)




                




if __name__ == "__main__":
    args = get_arguments()
    data_df = pd.read_pickle("../processed/base_data.pickle")
    rolling_feat_df = pd.read_pickle("../processed/Rolling_id_LAG_28_data.ftr")
    data_df = pd.concat([data_df, rolling_feat_df], axis=1)
    generate_features(globals(), args.force)