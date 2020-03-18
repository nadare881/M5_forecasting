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

class Calendar(Feature):
    def create_features(self):
        datas = []

        # 特徴量の作成
        calendar_df = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")
        calendar_df["date"] = pd.to_datetime(calendar_df["date"])
        calendar_df["day"] = calendar_df["date"].dt.day
        calendar_df["dayofyear"] = calendar_df["date"].dt.dayofyear
        calendar_df["d"] = calendar_df["d"].apply(lambda x: int(x[2:])).astype(np.int16)
        calendar_df = self.holiday_distance(calendar_df)

        # 型の圧縮
        intcol = ["wday", "month", "year", 'next_Sporting', 'next_Cultural','day', 'dayofyear']
        floatcol = ['last_Sporting', 'last_Cultural', 'last_National', 'last_Religious',]
        calendar_df.loc[:, intcol] = calendar_df.loc[:, intcol].astype(np.int16)
        calendar_df.loc[:, floatcol] = calendar_df.loc[:, floatcol].astype(np.float16)

        # マージ
        calendar_col = ['wm_yr_wk', 'wday', 'month', 'year', 'day', 'dayofyear', 'd', 'next_Sporting', 'last_Sporting',
                    'next_Cultural', 'last_Cultural', 'last_National', 'last_Religious']
        datas.append(data_df[["d"]].merge(calendar_df[calendar_col],  
                                          on="d",
                                          how="left")
                                   .drop("d", axis=1))
        
        # SNAPの情報を利用
        snap_df = self.make_snap(calendar_df)
        datas.append(data_df[["d", "state_id"]].merge(snap_df,
                                                      on=["d", "state_id"],
                                                      how="left")
                                               .drop(["d", "state_id"], axis=1))

        self.data = pd.concat(datas, axis=1)

    def search_last(self, X):
        now = 365
        res = []
        for  x in X:
            if x:
                now = 0
            else:
                now += 1
            if now >= 365:
                res.append(np.NaN)
            else:
                res.append(now)
        return res
    
    def search_next(self, X):
        now = 365
        res = []
        for  x in X[::-1]:
            if x:
                now = 0
            else:
                now += 1
            if now >= 365:
                res.append(np.NaN)
            else:
                res.append(now)
        return res[::-1]

    # TODO 未来の祝日情報の追加
    def holiday_distance(self, calendar_df):
        sporting = []
        cultural = []
        national = []
        religious= []
        for i, row in calendar_df.iterrows():
            D = defaultdict(int)
            if not pd.isna(row["event_type_1"]):
                D[row["event_type_1"]] = 1
            if not pd.isna(row["event_type_2"]):
                D[row["event_type_2"]] = 1
            sporting.append(D["Sporting"])
            cultural.append(D["Cultural"])
            national.append(D["National"])
            religious.append(D["Religious"])
        calendar_df["next_Sporting"] = self.search_next(sporting)
        calendar_df["last_Sporting"] = self.search_last(sporting)
        calendar_df["next_Cultural"] = self.search_next(cultural)
        calendar_df["last_Cultural"] = self.search_last(cultural)
        calendar_df["last_National"] = self.search_last(national)
        calendar_df["last_Religious"] = self.search_last(religious)
        return calendar_df
    
    def make_snap(self, calendar_df):
        snap_df = []
        state_id_map = {'CA': 0, 'TX': 1, 'WI': 2}
        for k, v in state_id_map.items():
            tmp_df = calendar_df[["d", f"snap_{k}"]].rename({f"snap_{k}": "snap"},axis=1)
            tmp_df["snap"] = tmp_df["snap"].astype(np.int8)
            tmp_df["state_id"] = v
            tmp_df["state_id"] = tmp_df["state_id"].astype(np.int8)
            snap_df.append(tmp_df)
        snap_df = pd.concat(snap_df)
        return snap_df

if __name__ == "__main__":
    args = get_arguments()
    data_df = pd.read_pickle("../processed/base_data.pickle")
    generate_features(globals(), args.force)
