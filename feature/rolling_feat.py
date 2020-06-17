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

LAG = 28

class Rolling_id(Feature):
    def create_features(self):
        self.add_prefix_name(f"LAG_{LAG}")
        datas = []
        feat = pd.DataFrame()

        # id
        feat[f"id_lag_{LAG}_rmean_1"] = data_df.groupby("id")["target"].transform(lambda x: x.shift(LAG))
        feat[f"id_lag_{LAG+1}_rmean_1"] = data_df.groupby("id")["target"].transform(lambda x: x.shift(LAG+1))
        feat[f"id_lag_{LAG+2}_rmean_1"] = data_df.groupby("id")["target"].transform(lambda x: x.shift(LAG+2))
        feat[f"id_lag_{LAG}_rmean_7"] = data_df.groupby("id")["target"].transform(lambda x: x.shift(LAG).rolling(7).mean())
        feat[f"id_lag_{LAG}_rmean_28"] = data_df.groupby("id")["target"].transform(lambda x: x.shift(LAG).rolling(28).mean())
        feat[f"id_lag_{LAG}_rmean_63"] = data_df.groupby("id")["target"].transform(lambda x: x.shift(LAG).rolling(63).mean())
        feat[f"id_lag_{LAG}_rmean_91"] = data_df.groupby("id")["target"].transform(lambda x: x.shift(LAG).rolling(91).mean())
        feat[f"id_lag_{LAG}_rmean_182"] = data_df.groupby("id")["target"].transform(lambda x: x.shift(LAG).rolling(182).mean())

        feat[f"id_lag_{LAG}_rmean_rate_7_28"] = feat[f"id_lag_{LAG}_rmean_7"] / (feat[f"id_lag_{LAG}_rmean_28"] + 1e-9)

        feat[f"id_lag_{LAG}_rstd_28"] = data_df.groupby("id")["target"].transform(lambda x: x.shift(LAG).rolling(28).std())
        feat[f"id_lag_{LAG}_rskew_28"] = data_df.groupby("id")["target"].transform(lambda x: x.shift(LAG).rolling(28).skew())
        feat[f"id_lag_{LAG}_rkurt_28"] = data_df.groupby("id")["target"].transform(lambda x: x.shift(LAG).rolling(28).kurt())

        feat[f"id_lag_{LAG}_rskew_364"] = data_df.groupby("id")["target"].transform(lambda x: x.shift(LAG).rolling(364).skew())
        feat[f"id_lag_{LAG}_rkurt_364"] = data_df.groupby("id")["target"].transform(lambda x: x.shift(LAG).rolling(364).kurt())

        feat[f"id_lag_1y_rmean_7"] = data_df.groupby("id")["target"].transform(lambda x: x.shift(364).rolling(7).mean())
        feat[f"id_lag_1y_rmean_28"] = data_df.groupby("id")["target"].transform(lambda x: x.shift(364).rolling(28).mean())
        feat[f"id_lag_1y_rmean_91"] = data_df.groupby("id")["target"].transform(lambda x: x.shift(364).rolling(91).mean())

        feat[f"id_lag_{LAG}_rmean_28_per_1y"] = feat[f"id_lag_{LAG}_rmean_28"] / data_df.groupby("id")["target"].transform(lambda x: x.shift(364 + 28).rolling(28).mean())
            
        datas.append(feat.copy().astype(np.float32))

        data_df["dd7"] = data_df["d"]//7
        data_df["dp7"] = data_df["d"]%7
        pivot = data_df.pivot_table(columns=["id", "dp7"],
                                    index="dd7",
                                    values="target",
                                    aggfunc="mean").shift(-(-LAG//7))
        res = pivot.rolling(4).mean().unstack().reset_index().rename({0: f"id_lag_{-(-LAG//7)}w_rmean_4w"}, axis=1)
        datas.append(data_df[["id", "dp7", "dd7"]].merge(res, on=["id", "dp7", "dd7"], how="left").drop(["id", "dp7", "dd7"], axis=1).astype(np.float32))
        res = pivot.rolling(8).mean().unstack().reset_index().rename({0: f"id_lag_{-(-LAG//7)}w_rmean_8w"}, axis=1)
        datas.append(data_df[["id", "dp7", "dd7"]].merge(res, on=["id", "dp7", "dd7"], how="left").drop(["id", "dp7", "dd7"], axis=1).astype(np.float32))
        res = pivot.rolling(12).mean().unstack().reset_index().rename({0: f"id_lag_{-(-LAG//7)}w_rmean_12w"}, axis=1)
        datas.append(data_df[["id", "dp7", "dd7"]].merge(res, on=["id", "dp7", "dd7"], how="left").drop(["id", "dp7", "dd7"], axis=1).astype(np.float32))

        pivot = data_df.pivot_table(columns=["item_id", "dp7"],
                                    index="dd7",
                                    values="target",
                                    aggfunc="mean").shift(-(-LAG//7))

        res = pivot.rolling(4).mean().unstack().reset_index().rename({0: f"item_lag_{-(-LAG//7)}w_rmean_4w"}, axis=1)
        datas.append(data_df[["item_id", "dp7", "dd7"]].merge(res, on=["item_id", "dp7", "dd7"], how="left").drop(["item_id", "dp7", "dd7"], axis=1).astype(np.float32))

        # item_id
        feat = pd.DataFrame()
        #count_ = data_df.groupby(["item_id", "d"])["id"].nunique().reset_index()
        sum_ = data_df.groupby(["item_id", "d"])["target"].sum().reset_index()
        feat["item_id"] = sum_["item_id"]
        feat["d"] = sum_["d"]
        feat["target"] = sum_["target"]# / count_["id"]
        feat = feat.sort_values(by="d")

        for day in [1, 7, 28, 91]:
            feat[f"item_id_lag_{LAG}_rmean_{day}"] = feat.groupby("item_id")["target"].transform(lambda x: x.shift(LAG).rolling(day).mean())
        
        feat[f"item_id_lag_364_rmean_7"] = feat.groupby("item_id")["target"].transform(lambda x: x.shift(364).rolling(7, center=True, min_periods=1).mean())
        feat[f"item_id_lag_{LAG}_rstd_28"] = feat.groupby("item_id")["target"].transform(lambda x: x.shift(LAG).rolling(28).std())
        feat[f"item_id_lag_{LAG}_rskew_28"] = feat.groupby("item_id")["target"].transform(lambda x: x.shift(LAG).rolling(28).skew())
        feat[f"item_id_lag_{LAG}_rkurt_28"] = feat.groupby("item_id")["target"].transform(lambda x: x.shift(LAG).rolling(28).kurt())

        datas.append(data_df[["item_id", "d"]].merge(feat.drop("target", axis=1),
                                                     on=["item_id", "d"],
                                                     how="left").drop(["item_id", "d"], axis=1).astype(np.float32))
        data_df["cum_scaled_target"] = data_df.groupby("id")["target"]\
                                              .transform(lambda x: (x - (x.cumsum() / (~x.isna()).cumsum()))
                                                                    / (np.sqrt(((x**2).cumsum() / ((~x.isna()).cumsum()) - (x.cumsum() / ((~x.isna()).cumsum()))**2)+1e-9))).astype(np.float32)
        #data_df["cum_scaled_target"] = data_df.groupby("id")["target"].transform(lambda x: (x - x.mean())/(x.std()+1e-9))
        for id_col in [["all_id"], ["store_id"], ["dept_id"], ["store_id", "dept_id"]]:
            pf = "_".join(id_col)
            feat = data_df.groupby(id_col + ["d"])["cum_scaled_target"].mean().reset_index().rename({0: "cum_scaled_target"}, axis=1)
            feat = feat.sort_values(by="d")

            feat[f"{pf}_lag_364_rmean_7"] = feat.groupby(id_col)["cum_scaled_target"].transform(lambda x: x.shift(364).rolling(7, center=True, min_periods=1).mean())
            feat[f"{pf}_lag_{LAG}_rmean_{1}"] = feat.groupby(id_col)["cum_scaled_target"].transform(lambda x: x.shift(LAG))
            feat[f"{pf}_lag_{LAG}_rmean_{7}"] = feat.groupby(id_col)["cum_scaled_target"].transform(lambda x: x.shift(LAG).rolling(7).mean())
            feat[f"{pf}_lag_{LAG}_rmean_{28}"] = feat.groupby(id_col)["cum_scaled_target"].transform(lambda x: x.shift(LAG).rolling(28).mean())
            feat[f"{pf}_lag_{LAG}_rmean_{91}"] = feat.groupby(id_col)["cum_scaled_target"].transform(lambda x: x.shift(LAG).rolling(91).mean())
            datas.append(data_df[id_col + ["d"]].merge(feat.drop("cum_scaled_target", axis=1),
                                                        on= id_col + ["d"],
                                                        how="left").drop(id_col + ["d"], axis=1).astype(np.float16))
            collect()
        
        data_df.loc[:, "store_dept_simple_mean"] = data_df.groupby(["store_id", "dept_id", "d"])["target"].transform("mean")
        data_df.loc[:, f"store_dept_simple_mean_lag_{LAG}"] = data_df.groupby(["id"])["store_dept_simple_mean"].transform(lambda x: x.shift(LAG))
        datas.append(data_df[[f"store_dept_simple_mean_lag_{LAG}"]])
        self.data = pd.concat(datas, axis=1).astype(np.float32)

if __name__ == "__main__":
    args = get_arguments()
    data_df = pd.read_pickle("../processed/base_data.pickle")
    data_df["all_id"] = 0
    generate_features(globals(), args.force)

