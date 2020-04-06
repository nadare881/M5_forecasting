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

class Price(Feature):
    def create_features(self):
        datas = []
        raw_train_df = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_validation.csv")
        price_df = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv")
        item_id_map = {c:i for i,c in enumerate(raw_train_df["item_id"].unique())}
        store_id_map = {c:i for i, c in enumerate(raw_train_df["store_id"].unique())}
        state_id_map = {c:i for i, c in enumerate(raw_train_df["state_id"].unique())}
        price_df["store_id"] = price_df["store_id"].map(store_id_map)
        price_df["item_id"] = price_df["item_id"].map(item_id_map)
        price_df["sell_price"] = price_df["sell_price"].astype(np.float32)
        
        # 特徴量の作成
        price_df = self.time_price_agg(price_df)
        price_df = price_df.merge(price_df.groupby(["store_id", "item_id"])["sell_price"]\
                                        .agg(["count", "nunique"])\
                                        .reset_index()\
                                        .rename({"count": "sell_count",
                                                "nunique": "price_nunique"}, axis=1),
                                on=["store_id", "item_id"],
                                how="left")
        
        # データの圧縮
        price_df.loc[:, price_df.select_dtypes(np.int64).columns] = price_df.loc[:, price_df.select_dtypes(np.int64).columns].astype(np.int16)
        price_df.loc[:, price_df.select_dtypes(np.float64).columns] = price_df.loc[:, price_df.select_dtypes(np.float64).columns].astype(np.float32)


        datas.append(data_df[["store_id", "item_id", "wm_yr_wk"]].merge(price_df,
                                                                        on=["store_id", "item_id", "wm_yr_wk"],
                                                                        how="left")\
                                                                 .drop(["store_id", "item_id", "wm_yr_wk"], axis=1))
        self.data = pd.concat(datas, axis=1)


    def time_price_agg(self, price_df):
        price_data = []
        rprice_data = []
        for (store_id, item_id), df in tqdm(price_df.groupby(["store_id", "item_id"])):
            now = df["sell_price"].values[0]
            min_ = now*1
            max_ = now*1
            rate = np.NaN
            last = np.NaN
            for wm_yr_wk, sell_price in zip(df["wm_yr_wk"], df["sell_price"]):
                if not pd.isna(last):
                    last += 1
                if now != sell_price:
                    min_ = min(min_, sell_price)
                    max_ = max(max_, sell_price)
                    rate = sell_price / now
                    last = 0
                now = sell_price

                price_data.append({"store_id": store_id,
                                "item_id": item_id,
                                "sell_price": sell_price,
                                "wm_yr_wk": wm_yr_wk,
                                "min_price": min_,
                                "max_price": max_,
                                "last_changed": last,
                                "change_rate": rate})
            r_rate = np.NaN
            next_ = np.NaN
            now = df["sell_price"].values[-1]
            for wm_yr_wk, sell_price in zip(df["wm_yr_wk"].values[::-1], df["sell_price"].values[::-1]):
                if not pd.isna(next_):
                    next_ += 1
                if now != sell_price:
                    rate = sell_price / now
                    next_ = 0
                now = sell_price
                rprice_data.append(({"store_id": store_id,
                                    "item_id": item_id,
                                    "wm_yr_wk": wm_yr_wk,
                                    "next_change": next_,
                                    "rchange_rate": rate}))

        price_df = pd.DataFrame(price_data)
        price_df = price_df.merge(pd.DataFrame(rprice_data),
                                on=["store_id", "item_id", "wm_yr_wk"],
                                how="left")
        price_df["cent"] = price_df["sell_price"] % 1.0
        price_df["sell_price_per_min_price"] = price_df["sell_price"] / price_df["min_price"]
        price_df["sell_price_per_max_price"] = price_df["sell_price"] / price_df["max_price"]
        return price_df
        


if __name__ == "__main__":
    args = get_arguments()
    data_df = pd.read_pickle("../processed/base_data.pickle")
    calendar_df = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")
    calendar_df["d"] = calendar_df["d"].apply(lambda x: int(x[2:])).astype(np.int16)
    data_df = data_df.merge(calendar_df[["d", "wm_yr_wk"]],
                            on="d",
                            how="left")


    generate_features(globals(), args.force)
