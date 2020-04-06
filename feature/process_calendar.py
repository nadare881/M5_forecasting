import datetime as dt
from datetime import datetime
from collections import defaultdict
import os
import re
import sys
from gc import collect
sys.path.append("../util")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from feature import Feature, get_arguments, generate_features
Feature.dir = "../processed"

from scipy.stats import ttest_ind_from_stats

def get_p(m1, s1, n1, m2, s2, n2):
    return ttest_ind_from_stats(m1, s1, n1, m2, s2, n2, equal_var=False)[1]

def get_near_dist(name1, dist1, name2, dist2):
    if (dist1 <= dist2):
        return dist1
    else:
        return dist2

def get_near_name(name1, dist1, name2, dist2):
    if (dist1 <= dist2):
        return name1
    else:
        return name2

class Calendar(Feature):
    def create_features(self):
        datas = []

        # 特徴量の作成
        calendar_df = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")
        calendar_df["date"] = pd.to_datetime(calendar_df["date"])
        calendar_df["day"] = calendar_df["date"].dt.day
        calendar_df["dayofyear"] = calendar_df["date"].dt.dayofyear
        calendar_df["weekofmonth"] = calendar_df["day"].apply(lambda x: x//7)
        calendar_df["d"] = calendar_df["d"].apply(lambda x: int(x[2:])).astype(np.int16)
        calendar_df = self.create_calendar_feature(calendar_df)

        # マージ
        calendar_merge_col = ['wday', 'month', 'year', 'd', 'day', 'dayofyear', "weekofmonth", 'last_Sporting_distance',
                            'last_Cultural_distance', 'last_National_distance',
                            'last_Religious_distance', 'next_Sporting_distance',
                            'next_Cultural_distance', 'next_National_distance',
                            'next_Religious_distance', 'nearest_Cultural_distance',
                            'nearest_National_distance', 'NBA_duration']
        name_col = ['last_Sporting_name', 'last_Cultural_name', 'last_National_name', 'last_Religious_name', 'next_Sporting_name',
                    'next_Cultural_name', 'next_National_name', 'next_Religious_name', 'nearest_Cultural_name', 'nearest_National_name']
        datas.append(data_df[["d"]].merge(calendar_df[calendar_merge_col + name_col],  
                                          on="d",
                                          how="left")
                                   .drop("d", axis=1))
                
        calendar_data = data_df.merge(calendar_df,  
                                    on="d",
                                    how="left")
        dept_data = data_df.query("target >= 0").groupby("dept_id")["target"].agg(["mean", "std", "count"])

        dept_merge_col = []

        res = calendar_data.pivot_table(columns=["id", "month", "day"],
                                        index="year",
                                        values="target",
                                        aggfunc="first")\
                            .shift(1)\
                            .unstack()\
                            .reset_index()\
                            .rename({0: "id_lastyear_daymonth"}, axis=1)
        calendar_data = calendar_data.merge(res,  
                                            on=["year", "id", "month", "day"],
                                            how="left")
        calendar_data["id_lastyear_daymonth"] = calendar_data["id_lastyear_daymonth"].astype(np.float16)    
        dept_merge_col.append("id_lastyear_daymonth")

        pivot = calendar_data.pivot_table(columns=["id", "month", "weekofmonth", "wday"],
                                        index="year",
                                        values="target",
                                        aggfunc="first")\
                            .shift(1)
        res = pivot.unstack()\
                   .reset_index()\
                   .rename({0: "id_lastyear_wdaywmonth"}, axis=1) 
        calendar_data = calendar_data.merge(res,  
                                            on=["year", "id", "month", "weekofmonth", "wday"],
                                            how="left")
        dept_merge_col.append("id_lastyear_wdaywmonth")
        calendar_data["all_id"] = 0
        for cols in [["id"], ["item_id"]]:
            name = "_".join(cols) + "_lastyear_wdaywmonth_cummean"
            pivot = calendar_data.pivot_table(columns= cols + ["month", "weekofmonth", "wday"],
                                                    index="year",
                                                    values="target",
                                                    aggfunc="sum")\
                                        .shift(1)
            pivot = pivot.cumsum() / (~pivot.isna()).cumsum()
            pivot = pivot.fillna(method="ffill")
            res = pivot.unstack()\
                    .reset_index()\
                    .rename({0: name}, axis=1)
            calendar_data = calendar_data.merge(res,  
                                                on=["year", "month", "weekofmonth", "wday"] + cols,
                                                how="left")        
            calendar_data[name] = calendar_data[name].astype(np.float32)
            dept_merge_col.append(name)
        
        calendar_data["cum_scaled_target"] = calendar_data.groupby("id")["target"]\
                                                          .transform(lambda x: (x - (x.cumsum() / (np.arange(len(x))+1)))
                                                                                / (np.sqrt(((x**2).cumsum() / np.arange(len(x))+1) - (x.cumsum() / (np.arange(len(x))+1))**2)+1e-18))
        for cols in [["store_id"], ["dept_id"], ["all_id"], ["store_id", "dept_id"]]:
            name = "_".join(cols) + "_lastyear_wdaywmonth_cummean"
            pivot = calendar_data.pivot_table(columns= cols + ["month", "weekofmonth", "wday"],
                                                    index="year",
                                                    values="cum_scaled_target",
                                                    aggfunc="mean")\
                                        .shift(1)
            pivot = pivot.cumsum() / (~pivot.isna()).cumsum()
            pivot = pivot.fillna(method="ffill")
            res = pivot.unstack()\
                    .reset_index()\
                    .rename({0: name}, axis=1)
            calendar_data = calendar_data.merge(res,  
                                                on=["year", "month", "weekofmonth", "wday"] + cols,
                                                how="left")        
            calendar_data[name] = calendar_data[name].astype(np.float32)
            dept_merge_col.append(name)
        datas.append(calendar_data[dept_merge_col])
         
        
        # SNAPの情報を利用
        snap_df = self.make_snap(calendar_df)        
        datas.append(data_df[["d", "state_id"]].merge(snap_df,
                                                      on=["d", "state_id"],
                                                      how="left")
                                               .drop(["d", "state_id"], axis=1))
        tmp_df = calendar_data.merge(snap_df,
                               on=["d", "state_id"],
                               how="left")
        pivot = tmp_df.query("target >= 0").pivot_table(columns="snap",
                                                        index=["item_id", "year"],
                                                        values="target",
                                                        aggfunc=["mean", "std", "count"])\
                                        .groupby("item_id")\
                                        .shift(1)
        pivot.columns = ["_".join(map(str, col)) for col in pivot.columns]
        pivot.loc[:, ["mean_0", "mean_1", "std_0", "std_1"]] = pivot.groupby("item_id")[["mean_0", "mean_1", "std_0", "std_1"]].transform(lambda x: x.cumsum() / np.arange(len(x)))
        pivot.loc[:, ["count_0", "count_1"]] = pivot.groupby("item_id")[["count_0", "count_1"]].transform(lambda x: x.cumsum())
        pivot = pivot[~pivot.isna().any(axis=1)]
        pivot["snap_ttest_pvalue"] = np.vectorize(get_p)(pivot["mean_0"], pivot["std_0"]+1e-9, pivot["count_0"], pivot["mean_1"], pivot["std_1"]+1e-9, pivot["count_1"])                                                  
        tmp_df = tmp_df.merge(pivot.reset_index()[["item_id", "year", "snap_ttest_pvalue"]],
                              on=["item_id", "year"],
                              how="left")
        tmp_df = tmp_df.rename({tmp_df.columns[-1]: 'snap_ttest_pvalue'}, axis=1)
        datas[-1]["snap_ttest_pvalue"] = np.log10(tmp_df["snap_ttest_pvalue"].values).astype(np.float32)
        self.data = pd.concat(datas, axis=1)

    def create_calendar_feature(self, calendar_df):
        last_event_name = {"National": "MartinLutherKingDay", "Cultural": "Halloween", "Sporting": "NBAFinalsEnd", "Religious": "OrthodoxChristmas"}
        last_event_date = {"National": datetime(2011, 1, 17), "Cultural": datetime(2010, 10, 31), "Sporting": datetime(2010, 6, 17), "Religious": datetime(2011, 1, 7)}
        last_event_date = {k: pd.datetime.date(v) for k, v in last_event_date.items()}

        feature = defaultdict(list)

        for i, row in calendar_df.iterrows():
            if not pd.isna(row["event_type_1"]):
                last_event_name[row["event_type_1"]] = row["event_name_1"]
                last_event_date[row["event_type_1"]] = pd.datetime.date(row["date"])
            if not pd.isna(row["event_type_2"]):
                last_event_name[row["event_type_2"]] = row["event_name_2"]
                last_event_date[row["event_type_2"]] = pd.datetime.date(row["date"])
            
            feature["d"].append(row["d"])
            feature["last_Sporting_name"].append(last_event_name["Sporting"])
            feature["last_Cultural_name"].append(last_event_name["Cultural"])
            feature["last_National_name"].append(last_event_name["National"])
            feature["last_Religious_name"].append(last_event_name["Religious"])
            
            feature["last_Sporting_distance"].append((pd.datetime.date(row["date"])-last_event_date["Sporting"]).days)
            feature["last_Cultural_distance"].append((pd.datetime.date(row["date"])-last_event_date["Cultural"]).days)
            feature["last_National_distance"].append((pd.datetime.date(row["date"])-last_event_date["National"]).days)
            feature["last_Religious_distance"].append((pd.datetime.date(row["date"])-last_event_date["Religious"]).days)
            
        calendar_df = calendar_df.merge(pd.DataFrame(feature),
                                        on="d",
                                        how="left")

        next_event_name = {"National": "IndependenceDay", "Religious": "Eid al-Fitr"}
        next_event_date = {"National": datetime(2016, 7, 4), "Religious": datetime(2016, 7, 6)}
        next_event_date = {k: pd.datetime.date(v) for k, v in next_event_date.items()}
        feature = defaultdict(list)

        for i, row in list(calendar_df.iterrows())[::-1]:
            if not pd.isna(row["event_type_1"]):
                next_event_name[row["event_type_1"]] = row["event_name_1"]
                next_event_date[row["event_type_1"]] = pd.datetime.date(row["date"])
            if not pd.isna(row["event_type_2"]):
                next_event_name[row["event_type_2"]] = row["event_name_2"]
                next_event_date[row["event_type_2"]] = pd.datetime.date(row["date"])
            
            feature["d"].append(row["d"])
            feature["next_Sporting_name"].append(next_event_name["Sporting"])
            feature["next_Cultural_name"].append(next_event_name["Cultural"])
            feature["next_National_name"].append(next_event_name["National"])
            feature["next_Religious_name"].append(next_event_name["Religious"])
            
            feature["next_Sporting_distance"].append((next_event_date["Sporting"]-pd.datetime.date(row["date"])).days)
            feature["next_Cultural_distance"].append((next_event_date["Cultural"]-pd.datetime.date(row["date"])).days)
            feature["next_National_distance"].append((next_event_date["National"]-pd.datetime.date(row["date"])).days)
            feature["next_Religious_distance"].append((next_event_date["Religious"]-pd.datetime.date(row["date"])).days)

        calendar_df = calendar_df.merge(pd.DataFrame(feature),
                                        on="d",
                                        how="left")

        calendar_df["nearest_Cultural_distance"] = np.vectorize(get_near_dist)(calendar_df["last_Cultural_name"], calendar_df["last_Cultural_distance"], calendar_df["next_Cultural_name"], calendar_df["next_Cultural_distance"])
        calendar_df["nearest_National_distance"] = np.vectorize(get_near_dist)(calendar_df["last_National_name"], calendar_df["last_National_distance"], calendar_df["next_National_name"], calendar_df["next_National_distance"])
        calendar_df["nearest_Cultural_name"] = np.vectorize(get_near_name)(calendar_df["last_Cultural_name"], calendar_df["last_Cultural_distance"], calendar_df["next_Cultural_name"], calendar_df["next_Cultural_distance"])
        calendar_df["nearest_National_name"] = np.vectorize(get_near_name)(calendar_df["last_National_name"], calendar_df["last_National_distance"], calendar_df["next_National_name"], calendar_df["next_National_distance"])

        calendar_df["NBA_duration"] = ((calendar_df["last_Sporting_name"] == "NBAFinalsStart") |
                                    ((calendar_df["last_Sporting_name"] == "NBAFinalsEnd") & ((calendar_df["last_Sporting_distance"] == 0)))).astype(np.uint8)

        event_name_map = {k: v for v, k in enumerate(set(calendar_df["event_name_1"].unique()[1:]) | set(calendar_df["event_name_2"].unique()[1:]))}
        name_col = ['last_Sporting_name', 'last_Cultural_name', 'last_National_name', 'last_Religious_name', 'next_Sporting_name',
                    'next_Cultural_name', 'next_National_name', 'next_Religious_name', 'nearest_Cultural_name', 'nearest_National_name']
        for col in name_col:
            calendar_df.loc[:, col] = calendar_df.loc[:, col].map(event_name_map).astype(np.uint8)
        dist_col = calendar_df.filter(regex="distance").columns
        calendar_df.loc[:, dist_col] = calendar_df.loc[:, dist_col].astype(np.int16)

        calendar_df[["wday", "month", "year", 'day', 'dayofyear', "weekofmonth"]] = calendar_df[["wday", "month", "year", 'day', 'dayofyear', "weekofmonth"]].astype(np.uint16)
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
