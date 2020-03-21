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

def rolling_fit(X, Y, window=28, th=10):
    
    def var(sum_, sum2, N):
        return max(sum2/N - (sum_/N)**2, 0)
    
    if type(X) == pd.Series:
        X = X.values
    if type(Y) == pd.Series:
        Y = Y.values
    
    QX = deque()
    QY = deque()
    res = defaultdict(lambda: [])
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
            res["slope"].append(np.NaN)
            res["pred"].append(np.NaN)
            res["R"].append(np.NaN)
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
            slope = covXY/VX
            intercept = Y_sum/N - slope*X_sum/N
            pred = (x+LAG)*slope + intercept
            r = covXY / np.sqrt(VX+1e-9) / np.sqrt(VY+1e-9)
            res["slope"].append(slope)
            res["pred"].append(pred)
            res["R"].append(r)
        else:
            res["slope"].append(np.NaN)
            res["pred"].append(np.NaN)
            res["R"].append(np.NaN)
    return res

class Rolling_linfit(Feature):
    def create_features(self):
        self.add_prefix_name(f"LAG_{LAG}")
        datas = []
        feat = pd.DataFrame()

        linfit_data = defaultdict(lambda : [])
        for id_, df in tqdm(data_df.groupby("id"), total=data_df["id"].nunique()):
            X = df["d"].shift(LAG)
            Y = df[f"id_lag_{LAG}_rmean_1"]
            Y2 = df[f"id_lag_{LAG}_rmean_91"]
            
            lfit_r1_w28 = rolling_fit(X, Y2, 28)
            lfit_r1_w91 = rolling_fit(X, Y, 91)
            lfit_r1_w364 = rolling_fit(X, Y, 364)
            lfit_r1_all = rolling_fit(X, Y, 10000)

            for i, d in enumerate(df["d"].values):
                linfit_data["id"].append(id_)
                linfit_data["d"].append(d)
                linfit_data["id_linfit_w28_slope"].append(lfit_r1_w28["slope"][i])
                linfit_data["id_linfit_w28_pred"].append(lfit_r1_w28["pred"][i])
                linfit_data["id_linfit_w28_r"].append(lfit_r1_w28["R"][i])

                linfit_data["id_linfit_w91_slope"].append(lfit_r1_w91["slope"][i])
                linfit_data["id_linfit_w91_pred"].append(lfit_r1_w91["pred"][i])
                linfit_data["id_linfit_w91_r"].append(lfit_r1_w91["R"][i])

                linfit_data["id_linfit_w364_slope"].append(lfit_r1_w364["slope"][i])
                linfit_data["id_linfit_w364_pred"].append(lfit_r1_w364["pred"][i])
                linfit_data["id_linfit_w364_r"].append(lfit_r1_w364["R"][i])

                linfit_data["id_linfit_wall_slope"].append(lfit_r1_all["slope"][i])
                linfit_data["id_linfit_wall_pred"].append(lfit_r1_all["pred"][i])
                linfit_data["id_linfit_wall_r"].append(lfit_r1_all["R"][i])
        linfit_data = pd.DataFrame(dict(linfit_data))
        datas.append(data_df[["id", "d"]].merge(linfit_data,
                                                on=["id", "d"],
                                                how="left")\
                                         .drop(["id", "d"], axis=1)\
                                         .astype(np.float32))

        del linfit_data
        collect()
        self.data = pd.concat(datas, axis=1)

if __name__ == "__main__":
    args = get_arguments()
    data_df = pd.read_pickle("../processed/base_data.pickle")
    rolling_feat_df = pd.read_pickle(f"../processed/Rolling_id_LAG_{LAG}_data.ftr")
    data_df = pd.concat([data_df, rolling_feat_df], axis=1)
    generate_features(globals(), args.force)