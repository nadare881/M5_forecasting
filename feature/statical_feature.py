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
from scipy.stats import skew, kurtosis

from feature import Feature, get_arguments, generate_features
Feature.dir = "../processed"

LAG = 28

class LastYear_stat(Feature):
    def create_features(self):
        data_df["target_r28"] = data_df.groupby("id")["target"].transform(lambda x: x.rolling(28, center=True, min_periods=1).mean())
        pivot = data_df.pivot_table(columns="dayofyear",
                                    index=["id", "year"],
                                    values="target_r28",
                                    aggfunc="first")\
                        .groupby("id")\
                        .shift(1)

        res = defaultdict(list)
        for (id_, year), row in tqdm(pivot.iterrows(), total=pivot.shape[0]):
            if (pd.isna(row[1])):
                continue
            if year == 2013:
                V = np.hstack([row[:359], row[360:]])
            else:
                V = np.hstack([row[:358], row[359:365]])
            center = np.argmax(V)
            
            res["id"].append(id_)
            res["year"].append(year)
            res["lastyear_min"].append(np.min(V))
            res["lastyear_max"].append(np.max(V))
            res["lastyear_argmin"].append(np.argmin(V))
            res["lastyear_argmax"].append(center)

        res_df = data_df[["id", "year", "dayofyear"]].merge(pd.DataFrame(res),
                                                            on=["id", "year"],
                                                            how="left")
        res_df["dist_from_argmax_lastyear"] = np.minimum((res_df["dayofyear"] - res_df["lastyear_argmax"])%364, (res_df["lastyear_argmax"] - res_df["dayofyear"])%364)
        res_df["dist_from_argmax_lastyear"] = np.minimum((res_df["dayofyear"] - res_df["lastyear_argmin"])%364, (res_df["lastyear_argmin"] - res_df["dayofyear"])%364)
        self.data = res_df.drop(["id", "year", "dayofyear"], axis=1).astype(np.float32)

class Ranking_feature(Feature):
    def create_features(self):
        self.add_prefix_name(f"LAG_{LAG}")
        roling_feat_df = pd.read_pickle(f"../processed/Rolling_id_LAG_{LAG}_data.ftr")[['id_lag_28_rmean_28']]
        feat_df = pd.concat([data_df, roling_feat_df], axis=1)
        feat_cols = []

        res = feat_df.pivot_table(index=["d", "store_id", "dept_id", "item_id"],
                                    values='id_lag_28_rmean_28',
                                    aggfunc="first")\
                    .groupby(level=[0, 1, 2])\
                    .rank(method="first", ascending=False)\
                    .stack()\
                    .reset_index()\
                    .rename({0: f"rank_store_dept_lag_{LAG}"}, axis=1)\
                    .drop("level_4", axis=1)
        feat_df = feat_df.merge(res,
                                on=["d", "store_id", "dept_id", "item_id"],
                                how="left")
        feat_df["target_rmean_7"] = feat_df.groupby("id")["target"].transform(lambda x: x.rolling(7, center=True).mean())     

        pivot = feat_df.pivot_table(columns=["store_id", "dept_id", f"rank_store_dept_lag_{LAG}"],
                                    index="d",
                                    values='target_rmean_7',
                                    aggfunc="first")\
                    .fillna(0)\
                    .shift(364)
        res = pivot.rolling(5, min_periods=1, center=True)\
                    .mean()\
                    .unstack()\
                    .reset_index()\
                    .rename({0: f"lastyear_rank_store_dept_lag_{LAG}"}, axis=1)
        feat_df = feat_df.merge(res,
                                on=["store_id", "dept_id", f"rank_store_dept_lag_{LAG}", "d"],
                                how="left")
        feat_cols.append(f"lastyear_rank_store_dept_lag_{LAG}")
        if (LAG < 0):
            for direction in ["last", "next"]:
                for event in ["National", "Cultural", "Sporting", "Religious"]:
                    feat_cols.append(f"{direction}_{event}_rank_store_dept_lag_24_win7")
                    res = feat_df.pivot_table(columns="year",
                                            index=["store_id", "dept_id", f"{direction}_{event}_distance", f"{direction}_{event}_name", f"rank_store_dept_lag_{LAG}"],
                                            values="target",
                                            aggfunc="mean")\
                                .fillna(0)\
                                .groupby(level=[0, 1, 2, 3])\
                                .rolling(3, center=True, min_periods=1)\
                                .mean()\
                                .shift(1, axis=1)
                    res.index = res.index.droplevel([0, 1, 2, 3])
                    feat_df = feat_df.merge(res.stack().reset_index().rename({0: f"{direction}_{event}_rank_store_dept_lag_24_win7"}, axis=1),
                                            on=["year", "store_id", "dept_id", f"{direction}_{event}_distance", f"{direction}_{event}_name", f"rank_store_dept_lag_{LAG}"],
                                            how="left")
                    print(feat_cols[-1])
        self.data = feat_df[feat_cols].astype(np.float32)

accum_add_prod = np.frompyfunc(lambda x, y: int((x+y)*y), 2, 1)

class Duration_stat(Feature):
    def create_features(self):
        self.add_prefix_name(f"LAG_{LAG}")
        dfs = []
        for id_, df in tqdm(data_df[["id", "d", "target"]].groupby("id"), total=30490):
            df["id_zerostreak"] = accum_add_prod.accumulate((df["target"]==0).astype(int), dtype=np.object)
            df[f"id_zerostreak_LAG_{LAG}"] = df["id_zerostreak"].shift(LAG)
            df["id_zerostreak_cummax"] = df["id_zerostreak"].shift(LAG).cummax()
            df["id_zerostreak_rmax_w364"] = df["id_zerostreak"].shift(LAG).rolling(364, min_periods=1).max() 
            df.loc[~(df["target"].shift(-1) > 0), "id_zerostreak"] = np.NaN
            df[f"id_zerostreak_mean_LAG_{LAG}_w365"] = df["id_zerostreak"].shift(LAG).rolling(365, min_periods=1).mean()
            df[f"id_zerostreak_std_LAG_{LAG}_w365"] = df["id_zerostreak"].shift(LAG).rolling(365, min_periods=1).std()
            df[f"id_zerostreak_skew_LAG_{LAG}_w365"] = df["id_zerostreak"].shift(LAG).rolling(365, min_periods=1).skew()
            df[f"id_zerostreak_kurt_LAG_{LAG}_w365"] = df["id_zerostreak"].shift(LAG).rolling(365, min_periods=1).kurt()
            dfs.append(df.drop(["target", "id_zerostreak"], axis=1))
        
        self.data = data_df[["id", "d"]].merge(pd.concat(dfs),
                                               on=["id", "d"],
                                               how="left")\
                                        .drop(["id", "d"], axis=1)\
                                        .astype(np.float32)




if __name__ == "__main__":
    args = get_arguments()
    data_df = pd.read_pickle("../processed/base_data.pickle")
    calendar_feat_df = pd.read_pickle("../processed/Calendar_data.ftr")
    data_df = pd.concat([data_df, calendar_feat_df], axis=1)
    data_df["all_id"] = 0
    generate_features(globals(), args.force)

