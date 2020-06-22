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

from scipy.stats import norm

from tqdm import tqdm
from joblib import Parallel, delayed

from feature import Feature, get_arguments, generate_features

from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation

import logging
logging.getLogger('fbprophet').setLevel(logging.WARNING)

Feature.dir = "../processed"

LAG = 7
class Prophet_id_oof(Feature):
    def create_features(self):
        data_df = pd.read_pickle("../processed/base_data.pickle")
        id_prophet = pd.read_csv("../prophets/id_prophet_result.csv")

        raw_train_df = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_evaluation.csv")

        id_map = {v:i for i, v in enumerate(raw_train_df["id"].values)}

        id_prophet["yhat"] = id_prophet["yhat"].astype(np.float16)
        id_prophet["yhat_0.25"] = id_prophet["yhat_0.25"].astype(np.float16)
        id_prophet["yhat_0.75"] = id_prophet["yhat_0.75"].astype(np.float16)

        id_prophet = id_prophet.rename({"yhat": "id_yhat", "yhat_0.25": "id_yhat_0.25", "yhat_0.75": "id_yhat_0.75"}, axis=1)
        id_prophet["id_yhat_scale"] = id_prophet["id_yhat_0.75"] - id_prophet["id_yhat_0.25"]
        id_prophet["id_yhat_scale_per_loc"] = id_prophet["id_yhat_scale"] / id_prophet["id_yhat"]
        id_prophet["id"] = id_prophet["id"].map(id_map)
        self.data = data = data_df.merge(id_prophet,
                                         on=["id", "d"],
                                         how="left").filter(regex="id_yhat")


class Prophet_item_oof(Feature):
    def create_features(self):
        data_df = pd.read_pickle("../processed/base_data.pickle")
        id_prophet = pd.read_csv("../prophets/id_prophet_result.csv")

        raw_train_df = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_evaluation.csv")
        
        item_prophet = pd.read_csv("../prophets/item_prophet_result.csv")
        item_prophet = item_prophet.rename({col: "item_" + col for col in item_prophet.filter(regex="yhat_").columns}, axis=1)
        item_prophet["item_yhat_scale"] = ((item_prophet["item_yhat_0.995"] - item_prophet["item_yhat_0.005"])/norm().ppf(0.995) +
                                        (item_prophet["item_yhat_0.975"] - item_prophet["item_yhat_0.025"])/norm().ppf(0.975) +
                                        (item_prophet["item_yhat_0.835"] - item_prophet["item_yhat_0.165"])/norm().ppf(0.835) +
                                        (item_prophet["item_yhat_0.750"] - item_prophet["item_yhat_0.250"])/norm().ppf(0.75))/4
        item_prophet["item_yhat_scale_per_loc"] = item_prophet["item_yhat_scale"] / item_prophet["item_yhat_0.500"]
        for col in item_prophet.filter(regex="item_yhat_").columns:
            item_prophet[col] = item_prophet[col].astype(np.float16)
        item_id_map = {c:i for i,c in enumerate(raw_train_df["item_id"].unique())}
        item_prophet["item_id"] = item_prophet["item_id"].map(item_id_map)
        self.data = data_df.merge(item_prophet,
                                  on=["item_id", "d"],
                                  how="left").filter(regex="item_yhat")

class Prophet_level_oof(Feature):
    def create_features(self):
        self.add_prefix_name(f"LAG_{LAG}")
        raw_train_df = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_evaluation.csv")

        data_df = pd.read_pickle("../processed/base_data.pickle")
        prophet_df = pd.read_csv(f"../prophets/lag{LAG}_level9_prophet.csv")
        prophet_df["id"] = prophet_df["id"] + "_evaluation"
        level_df = pd.read_pickle("../prophets/level1_9.pkl")

        store_id_map = {c:i for i, c in enumerate(raw_train_df["store_id"].unique())}
        state_id_map = {c:i for i, c in enumerate(raw_train_df["state_id"].unique())}
        level_df["dept_id"] = level_df["dept_id"].map({c:i for i, c in enumerate(raw_train_df["dept_id"].unique())}).fillna(-1).astype(np.int8)
        level_df["cat_id"] = level_df["cat_id"].map({c:i for i, c in enumerate(raw_train_df["cat_id"].unique())}).fillna(-1).astype(np.int8)
        level_df["store_id"] = level_df["store_id"].map(store_id_map).fillna(-1).astype(np.int8)
        level_df["state_id"] = level_df["state_id"].map(state_id_map).fillna(-1).astype(np.int8)

        prophet_df = prophet_df.merge(level_df[["d", "id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "level", "record_count"]],
                                    on=["id", "d"],
                                    how="left")
        for col in prophet_df.filter(regex="yhat_").columns:
            prophet_df[col] = prophet_df[col] / prophet_df["record_count"]

        data = []

        #level 1
        tmp_df = prophet_df.query("level == 1")
        pf = "all_id_"

        tmp_df[pf + "yhat"] = tmp_df["yhat_0.500"].astype(np.float16)
        tmp_df[pf + "scale"] = (((tmp_df["yhat_0.995"] - tmp_df["yhat_0.005"])/norm().ppf(0.995) +
                                (tmp_df["yhat_0.975"] - tmp_df["yhat_0.025"])/norm().ppf(0.975) +
                                (tmp_df["yhat_0.835"] - tmp_df["yhat_0.165"])/norm().ppf(0.835) +
                                (tmp_df["yhat_0.750"] - tmp_df["yhat_0.250"])/norm().ppf(0.75))/4).astype(np.float16)
        tmp_df[pf + "scale_per_loc"] = tmp_df[pf + "yhat"] / tmp_df[pf + "scale"] 
        data.append(data_df[["d"]].merge(tmp_df,
                                        on="d",
                                        how="left").filter(regex=pf))

        #level 2
        tmp_df = prophet_df.query("level == 2")
        pf = "state_X_"
        mcol = ["d", "state_id"]
        cols = mcol + [pf + "yhat", pf + "scale", pf + "scale_per_loc"]
        tmp_df[pf + "yhat"] = tmp_df["yhat_0.500"].astype(np.float16)
        tmp_df[pf + "scale"] = (((tmp_df["yhat_0.995"] - tmp_df["yhat_0.005"])/norm().ppf(0.995) +
                                (tmp_df["yhat_0.975"] - tmp_df["yhat_0.025"])/norm().ppf(0.975) +
                                (tmp_df["yhat_0.835"] - tmp_df["yhat_0.165"])/norm().ppf(0.835) +
                                (tmp_df["yhat_0.750"] - tmp_df["yhat_0.250"])/norm().ppf(0.75))/4).astype(np.float16)
        tmp_df[pf + "scale_per_loc"] = tmp_df[pf + "yhat"] / tmp_df[pf + "scale"] 
        data.append(data_df[mcol].merge(tmp_df[cols],
                                        on=mcol,
                                        how="left").filter(regex=pf))

        #level 3
        tmp_df = prophet_df.query("level == 3")
        pf = "store_X_"
        mcol = ["d", "store_id"]
        cols = mcol + [pf + "yhat", pf + "scale", pf + "scale_per_loc"]
        tmp_df[pf + "yhat"] = tmp_df["yhat_0.500"].astype(np.float16)
        tmp_df[pf + "scale"] = (((tmp_df["yhat_0.995"] - tmp_df["yhat_0.005"])/norm().ppf(0.995) +
                                (tmp_df["yhat_0.975"] - tmp_df["yhat_0.025"])/norm().ppf(0.975) +
                                (tmp_df["yhat_0.835"] - tmp_df["yhat_0.165"])/norm().ppf(0.835) +
                                (tmp_df["yhat_0.750"] - tmp_df["yhat_0.250"])/norm().ppf(0.75))/4).astype(np.float16)
        tmp_df[pf + "scale_per_loc"] = tmp_df[pf + "yhat"] / tmp_df[pf + "scale"] 
        data.append(data_df[mcol].merge(tmp_df[cols],
                                        on=mcol,
                                        how="left").filter(regex=pf)) 

        #level 4
        tmp_df = prophet_df.query("level == 4")
        pf = "cat_X_"
        mcol = ["d", "cat_id"]
        cols = mcol + [pf + "yhat", pf + "scale", pf + "scale_per_loc"]
        tmp_df[pf + "yhat"] = tmp_df["yhat_0.500"].astype(np.float16)
        tmp_df[pf + "scale"] = (((tmp_df["yhat_0.995"] - tmp_df["yhat_0.005"])/norm().ppf(0.995) +
                                (tmp_df["yhat_0.975"] - tmp_df["yhat_0.025"])/norm().ppf(0.975) +
                                (tmp_df["yhat_0.835"] - tmp_df["yhat_0.165"])/norm().ppf(0.835) +
                                (tmp_df["yhat_0.750"] - tmp_df["yhat_0.250"])/norm().ppf(0.75))/4).astype(np.float16)
        tmp_df[pf + "scale_per_loc"] = tmp_df[pf + "yhat"] / tmp_df[pf + "scale"] 
        data.append(data_df[mcol].merge(tmp_df[cols],
                                        on=mcol,
                                        how="left").filter(regex=pf))

        #level 5
        tmp_df = prophet_df.query("level == 5")
        pf = "dept_X_"
        mcol = ["d", "dept_id"]
        cols = mcol + [pf + "yhat", pf + "scale", pf + "scale_per_loc"]
        tmp_df[pf + "yhat"] = tmp_df["yhat_0.500"].astype(np.float16)
        tmp_df[pf + "scale"] = (((tmp_df["yhat_0.995"] - tmp_df["yhat_0.005"])/norm().ppf(0.995) +
                                (tmp_df["yhat_0.975"] - tmp_df["yhat_0.025"])/norm().ppf(0.975) +
                                (tmp_df["yhat_0.835"] - tmp_df["yhat_0.165"])/norm().ppf(0.835) +
                                (tmp_df["yhat_0.750"] - tmp_df["yhat_0.250"])/norm().ppf(0.75))/4).astype(np.float16)
        tmp_df[pf + "scale_per_loc"] = tmp_df[pf + "yhat"] / tmp_df[pf + "scale"] 
        data.append(data_df[mcol].merge(tmp_df[cols],
                                        on=mcol,
                                        how="left").filter(regex=pf))

        #level 6
        tmp_df = prophet_df.query("level == 6")
        pf = "state_cat_"
        mcol = ["d", "state_id", "cat_id"]
        cols = mcol + [pf + "yhat", pf + "scale", pf + "scale_per_loc"]
        tmp_df[pf + "yhat"] = tmp_df["yhat_0.500"].astype(np.float16)
        tmp_df[pf + "scale"] = (((tmp_df["yhat_0.995"] - tmp_df["yhat_0.005"])/norm().ppf(0.995) +
                                (tmp_df["yhat_0.975"] - tmp_df["yhat_0.025"])/norm().ppf(0.975) +
                                (tmp_df["yhat_0.835"] - tmp_df["yhat_0.165"])/norm().ppf(0.835) +
                                (tmp_df["yhat_0.750"] - tmp_df["yhat_0.250"])/norm().ppf(0.75))/4).astype(np.float16)
        tmp_df[pf + "scale_per_loc"] = tmp_df[pf + "yhat"] / tmp_df[pf + "scale"] 
        data.append(data_df[mcol].merge(tmp_df[cols],
                                        on=mcol,
                                        how="left").filter(regex=pf))

        #level 7
        tmp_df = prophet_df.query("level == 7")
        pf = "state_dept_"
        mcol = ["d", "state_id", "dept_id"]
        cols = mcol + [pf + "yhat", pf + "scale", pf + "scale_per_loc"]
        tmp_df[pf + "yhat"] = tmp_df["yhat_0.500"].astype(np.float16)
        tmp_df[pf + "scale"] = (((tmp_df["yhat_0.995"] - tmp_df["yhat_0.005"])/norm().ppf(0.995) +
                                (tmp_df["yhat_0.975"] - tmp_df["yhat_0.025"])/norm().ppf(0.975) +
                                (tmp_df["yhat_0.835"] - tmp_df["yhat_0.165"])/norm().ppf(0.835) +
                                (tmp_df["yhat_0.750"] - tmp_df["yhat_0.250"])/norm().ppf(0.75))/4).astype(np.float16)
        tmp_df[pf + "scale_per_loc"] = tmp_df[pf + "yhat"] / tmp_df[pf + "scale"] 
        data.append(data_df[mcol].merge(tmp_df[cols],
                                        on=mcol,
                                        how="left").filter(regex=pf))

        #level 8
        tmp_df = prophet_df.query("level == 8")
        pf = "store_cat_"
        mcol = ["d", "store_id", "cat_id"]
        cols = mcol + [pf + "yhat", pf + "scale", pf + "scale_per_loc"]
        tmp_df[pf + "yhat"] = tmp_df["yhat_0.500"].astype(np.float16)
        tmp_df[pf + "scale"] = (((tmp_df["yhat_0.995"] - tmp_df["yhat_0.005"])/norm().ppf(0.995) +
                                (tmp_df["yhat_0.975"] - tmp_df["yhat_0.025"])/norm().ppf(0.975) +
                                (tmp_df["yhat_0.835"] - tmp_df["yhat_0.165"])/norm().ppf(0.835) +
                                (tmp_df["yhat_0.750"] - tmp_df["yhat_0.250"])/norm().ppf(0.75))/4).astype(np.float16)
        tmp_df[pf + "scale_per_loc"] = tmp_df[pf + "yhat"] / tmp_df[pf + "scale"] 
        data.append(data_df[mcol].merge(tmp_df[cols],
                                        on=mcol,
                                        how="left").filter(regex=pf))

        #level 9
        tmp_df = prophet_df.query("level == 9")
        pf = "store_dept_"
        mcol = ["d", "store_id", "dept_id"]
        cols = mcol + [pf + "yhat", pf + "scale", pf + "scale_per_loc"]
        tmp_df[pf + "yhat"] = tmp_df["yhat_0.500"].astype(np.float16)
        tmp_df[pf + "scale"] = (((tmp_df["yhat_0.995"] - tmp_df["yhat_0.005"])/norm().ppf(0.995) +
                                (tmp_df["yhat_0.975"] - tmp_df["yhat_0.025"])/norm().ppf(0.975) +
                                (tmp_df["yhat_0.835"] - tmp_df["yhat_0.165"])/norm().ppf(0.835) +
                                (tmp_df["yhat_0.750"] - tmp_df["yhat_0.250"])/norm().ppf(0.75))/4).astype(np.float16)
        tmp_df[pf + "scale_per_loc"] = tmp_df[pf + "yhat"] / tmp_df[pf + "scale"] 
        data.append(data_df[mcol].merge(tmp_df[cols],
                                        on=mcol,
                                        how="left").filter(regex=pf))
        self.data = pd.concat(data, axis=1)

if __name__ == "__main__":
    args = get_arguments()
    generate_features(globals(), args.force)