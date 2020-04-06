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
from joblib import Parallel, delayed

from feature import Feature, get_arguments, generate_features

from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation

import logging
logging.getLogger('fbprophet').setLevel(logging.WARNING)

Feature.dir = "../processed"

LAG = 28

class Prophet_oof(Feature):
    def create_features(self):
        self.add_prefix_name(f"LAG_{LAG}")
        datas = []
        data_df = pd.read_pickle("../processed/base_data.pickle")
        data_df["all_id"] = 0
        calendar_df = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")
        calendar_df["date"] = pd.to_datetime(calendar_df["date"])
        calendar_df["d"] = calendar_df["d"].apply(lambda x: int(x[2:]))
        
        data_df = data_df.merge(calendar_df[["date", "d"]],
                                on="d",
                                how="left")
        datas = []
        for cols in [["all_id"], ["store_id"], ["dept_id"], ["dept_id", "store_id"]]:
            name = "_".join(cols)
            df = pd.read_csv(f"../download/{name}_prophet.csv").rename({name + "_" + "cutoff": "cutoff"}, axis=1)
            df["ds"] = pd.to_datetime(df["ds"])
            df["cutoff"] = pd.to_datetime(df["cutoff"])

            res = []
            res_cols = [name + "_" + c for c in ["yhat", "yhat_lower", "yhat_upper"]] + cols + ["ds"]
            ix = (np.timedelta64(LAG, 'D') < df["ds"] - df["cutoff"])&(df["ds"] - df["cutoff"] <= np.timedelta64(LAG+7, 'D'))
            res.append(df.loc[ix, res_cols])
            res.append(df[res_cols].loc[df["cutoff"].isna()])
            
            datas.append(data_df.merge(pd.concat(res).rename({"ds": "date"}, axis=1),
                                    on=["date"] + cols,
                                    how="left")[res_cols[:3]])
        self.data = pd.concat(datas, axis=1).astype(np.float32)

class Prophet_Item_oof(Feature):
    def create_features(self):
        data_df = pd.read_pickle("../processed/base_data.pickle")
        data_df["all_id"] = 0
        calendar_df = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")
        calendar_df["date"] = pd.to_datetime(calendar_df["date"])
        calendar_df["d"] = calendar_df["d"].apply(lambda x: int(x[2:]))
        
        data_df = data_df.merge(calendar_df[["date", "d"]],
                                on="d",
                                how="left")
        datas = []
        cols = ["item_id"]
        name = "_".join(cols)
        df = pd.concat([pd.read_csv(f"../download/item_id_prophet{i}.csv") for i in range(20)]).rename({name + "_" + "cutoff": "cutoff"}, axis=1)
        df["ds"] = pd.to_datetime(df["ds"])
        df["cutoff"] = pd.to_datetime(df["cutoff"])

        res = []
        res_cols = [name + "_" + c for c in ["yhat", "yhat_lower", "yhat_upper"]] + cols + ["ds"]
        ix = (np.timedelta64(28, 'D') < df["ds"] - df["cutoff"])
        res.append(df.loc[ix, res_cols])
        res.append(df[res_cols].loc[df["cutoff"].isna()])
        self.data = data_df.merge(pd.concat(res).rename({"ds": "date"}, axis=1),
                                  on=["date"] + cols,
                                  how="left")[res_cols[:3]].astype(np.float32)


if __name__ == "__main__":
    args = get_arguments()
    generate_features(globals(), args.force)