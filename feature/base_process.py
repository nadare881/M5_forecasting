import os
import datetime
import re
import sys
sys.path.append("../util")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
tqdm.pandas()

from feature import Feature, get_arguments, generate_features
Feature.dir = "../processed"

class Base(Feature):
    def create_features(self):
        feat = pd.DataFrame()
        feat["first_appear"] = data_df["first_appear"].replace(1, np.NaN).astype(np.float32)
        feat["since_appear"] = (data_df["d"] - data_df["first_appear"]).astype(np.float32)
        self.data = feat

if __name__ == "__main__":
    args = get_arguments()
    calendar_df = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")
    raw_train_df = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_validation.csv")
    price_df = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv")
    smpsb_df = pd.read_csv("../input/m5-forecasting-accuracy/sample_submission.csv")

    # idをintに
    item_id_map = {c:i for i,c in enumerate(raw_train_df["item_id"].unique())}
    store_id_map = {c:i for i, c in enumerate(raw_train_df["store_id"].unique())}
    state_id_map = {c:i for i, c in enumerate(raw_train_df["state_id"].unique())}
    raw_train_df["id"] = raw_train_df.index.astype(np.int16)
    raw_train_df["item_id"] = raw_train_df["item_id"].map(item_id_map).astype(np.int16)
    raw_train_df["dept_id"] = raw_train_df["dept_id"].map({c:i for i, c in enumerate(raw_train_df["dept_id"].unique())}).astype(np.int8)
    raw_train_df["cat_id"] = raw_train_df["cat_id"].map({c:i for i, c in enumerate(raw_train_df["cat_id"].unique())}).astype(np.int8)
    raw_train_df["store_id"] = raw_train_df["store_id"].map(store_id_map).astype(np.int8)
    raw_train_df["state_id"] = raw_train_df["state_id"].map(state_id_map).astype(np.int8)

    # 縦にデータを繋げる
    data = []
    for col in tqdm(calendar_df["d"].unique()):
        if (int(col[2:]) <= 1913):
            tmp_df = raw_train_df[list(raw_train_df.columns[:6]) + [col]].rename({col:"target"}, axis=1)
            tmp_df["target"] = tmp_df["target"].astype(np.int16)
        else:
            tmp_df = raw_train_df[list(raw_train_df.columns[:6])]
            tmp_df["target"] = -1
            tmp_df["target"] = tmp_df["target"].astype(np.int16)
        tmp_df["d"] = int(col[2:])
        tmp_df["d"] = tmp_df["d"].astype(np.int16)
        data.append(tmp_df)
    data_df = pd.concat(data)

    # 日付情報の取得
    calendar_df["date"] = pd.to_datetime(calendar_df["date"])
    calendar_df["day"] = calendar_df["date"].dt.day
    calendar_df["d"] = calendar_df["d"].apply(lambda x: int(x[2:])).astype(np.int16)
    data_df = data_df.merge(calendar_df[["d", "month", "day", "wm_yr_wk"]],
                            on="d",
                            how="left")
    
    # 商品に値段がついてないデータを除く
    # 年1の閉店日であるクリスマスのデータを除く
    price_df["store_id"] = price_df["store_id"].map(store_id_map)
    price_df["item_id"] = price_df["item_id"].map(item_id_map)
    data_df = data_df.merge(price_df,
                            on=["item_id", "store_id", "wm_yr_wk"],
                            how="left")
    data_df = data_df[~data_df["sell_price"].isna()]
    data_df = data_df.query("not (month == 12 and day == 25)").reset_index(drop=True)
    data_df["first_appear"] = data_df.groupby("id")["d"].transform("min")
    pd.to_pickle(data_df[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'target', 'd']], "../processed/base_data.pickle")
    generate_features(globals(), args.force)
