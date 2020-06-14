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


if __name__ == "__main__":
    args = get_arguments()
    data_df = pd.read_pickle("../processed/base_data.pickle")
    calendar_feat_df = pd.read_pickle("../processed/Calendar_data.ftr")
    data_df = pd.concat([data_df, calendar_feat_df], axis=1)
    data_df["all_id"] = 0
    generate_features(globals(), args.force)

