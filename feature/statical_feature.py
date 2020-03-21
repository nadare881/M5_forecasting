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

class LastYear_stat(Feature):
    def create_features(self):
        data_df["target_r28"] = data_df.groupby("id")["target"].transform(lambda x: x.rolling(28, center=True).mean())
        pivot = data_df.pivot_table(columns="dayofyear",
                                    index=["id", "year"],
                                    values="target_r28",
                                    aggfunc="first")\
                        .fillna(method="ffill", axis=1)\
                        .groupby("id")\
                        .shift(1)

        res = defaultdict(list)
        for (id_, year), row in tqdm(pivot.iterrows()):
            if (row.isna().sum() > 0):
                continue
            res["id"].append(id_)
            res["year"].append(year)
            res["lastyear_min"].append(np.min(row))
            res["lastyear_max"].append(np.max(row))
            res["lastyear_argmin"].append(np.argmin(row))
            res["lastyear_argmax"].append(np.argmax(row))

        res_df = data_df[["id", "year", "dayofyear"]].merge(pd.DataFrame(res),
                                                            on=["id", "year"],
                                                            how="left")
        res_df["dist_from_argmax_lastyear"] = np.minimum((res_df["dayofyear"] - res_df["lastyear_argmax"])%366, (res_df["lastyear_argmax"] - res_df["dayofyear"])%366)
        res_df["dist_from_argmax_lastyear"] = np.minimum((res_df["dayofyear"] - res_df["lastyear_argmin"])%366, (res_df["lastyear_argmin"] - res_df["dayofyear"])%366)
        self.data = res_df.drop(["id", "year", "dayofyear"], axis=1).astype(np.float32)


if __name__ == "__main__":
    args = get_arguments()
    data_df = pd.read_pickle("../processed/base_data.pickle")
    calendar_feat_df = pd.read_pickle("../processed/Calendar_data.ftr")
    data_df = pd.concat([data_df, calendar_feat_df], axis=1)
    data_df["all_id"] = 0
    generate_features(globals(), args.force)

