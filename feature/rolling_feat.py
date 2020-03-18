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
        feat[f"id_lag_{LAG}_rmean_7"] = data_df.groupby("id")["target"].transform(lambda x: x.shift(LAG).rolling(7).mean())
        feat[f"id_lag_{LAG}_rmean_28"] = data_df.groupby("id")["target"].transform(lambda x: x.shift(LAG).rolling(28).mean())
        feat[f"id_lag_{LAG}_rmean_91"] = data_df.groupby("id")["target"].transform(lambda x: x.shift(LAG).rolling(91).mean())

        feat[f"id_lag_{LAG}_rstd_28"] = data_df.groupby("id")["target"].transform(lambda x: x.shift(LAG).rolling(28).std())
        feat[f"id_lag_{LAG}_rskew_28"] = data_df.groupby("id")["target"].transform(lambda x: x.shift(LAG).rolling(28).skew())
        feat[f"id_lag_{LAG}_rkurt_28"] = data_df.groupby("id")["target"].transform(lambda x: x.shift(LAG).rolling(28).kurt())

        feat[f"id_lag_1y_rmean_7"] = data_df.groupby("id")["target"].transform(lambda x: x.shift(364).rolling(7).mean())
        feat[f"id_lag_1y_rmean_28"] = data_df.groupby("id")["target"].transform(lambda x: x.shift(364).rolling(28).mean())
        feat[f"id_lag_1y_rmean_91"] = data_df.groupby("id")["target"].transform(lambda x: x.shift(364).rolling(91).mean())

        feat[f"id_lag_{LAG}_rmean_28_per_1y"] = feat[f"id_lag_{LAG}_rmean_28"] / data_df.groupby("id")["target"].transform(lambda x: x.shift(364 + 28).rolling(28).mean())

        datas.append(feat.copy().astype(np.float32))

        # item_id
        feat = pd.DataFrame()
        count_ = data_df.groupby(["item_id", "d"])["id"].nunique().reset_index()
        sum_ = data_df.groupby(["item_id", "d"])["target"].nunique().reset_index()
        feat["item_id"] = count_["item_id"]
        feat["d"] = count_["d"]
        feat["target"] = sum_["target"] / count_["id"]
        feat = feat.sort_values(by="d")

        for day in [1, 7, 28, 91]:
            feat[f"item_id_lag_{LAG}_rmean_{day}"] = feat.groupby("item_id")["target"].transform(lambda x: x.shift(LAG).rolling(day).mean())

        feat[f"item_id_lag_{LAG}_rstd_28"] = feat.groupby("item_id")["target"].transform(lambda x: x.shift(LAG).rolling(28).std())
        feat[f"item_id_lag_{LAG}_rskew_28"] = feat.groupby("item_id")["target"].transform(lambda x: x.shift(LAG).rolling(28).skew())
        feat[f"item_id_lag_{LAG}_rkurt_28"] = feat.groupby("item_id")["target"].transform(lambda x: x.shift(LAG).rolling(28).kurt())

        datas.append(data_df[["item_id", "d"]].merge(feat.drop("target", axis=1),
                                                     on=["item_id", "d"],
                                                     how="left").drop(["item_id", "d"], axis=1).astype(np.float32))
            
        for id_col in [["all_id"], ["store_id"], ["dept_id"], ["store_id", "dept_id"]]:
            pf = "_".join(id_col)
            feat = pd.DataFrame()
            count_ = data_df.groupby(id_col + ["d"])["id"].nunique().reset_index()
            sum_ = data_df.groupby(id_col + ["d"])["target"].nunique().reset_index()
            for col in id_col:
                feat[col] = count_[col]
            feat["d"] = count_["d"]
            feat["target"] = sum_["target"] / count_["id"]
            feat = feat.sort_values(by="d")

            feat[f"{pf}_lag_{LAG}_rmean_{28}"] = feat.groupby(id_col)["target"].transform(lambda x: x.shift(LAG).rolling(28).mean())
            feat[f"{pf}_id_lag_{LAG}_rmean_{91}"] = feat.groupby(id_col)["target"].transform(lambda x: x.shift(LAG).rolling(91).mean())
            datas.append(data_df[id_col + ["d"]].merge(feat.drop("target", axis=1),
                                                        on= id_col + ["d"],
                                                        how="left").drop(id_col + ["d"], axis=1).astype(np.float32))
            collect()

        self.data = pd.concat(datas, axis=1).astype(np.float32)

if __name__ == "__main__":
    args = get_arguments()
    data_df = pd.read_pickle("../processed/base_data.pickle")
    data_df["all_id"] = 0
    generate_features(globals(), args.force)

