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
        calendar_df["d"] = calendar_df["d"].apply(lambda x: int(x[2:])).astype(np.int16)
        calendar_df = self.create_calendar_feature(calendar_df)

        # マージ
        calendar_merge_col = ['wday', 'month', 'year', 'd', 'day', 'dayofyear', 'last_Sporting_distance',
                            'last_Cultural_distance', 'last_National_distance',
                            'last_Religious_distance', 'next_Sporting_distance',
                            'next_Cultural_distance', 'next_National_distance',
                            'next_Religious_distance', 'nearest_Cultural_distance',
                            'nearest_National_distance', 'NBA_duration']
        datas.append(data_df[["d"]].merge(calendar_df[calendar_merge_col],  
                                          on="d",
                                          how="left")
                                   .drop("d", axis=1))
                
        calendar_data = data_df.merge(calendar_df,  
                                    on="d",
                                    how="left")
        dept_data = data_df.query("target >= 0").groupby("dept_id")["target"].agg(["mean", "std", "count"])

        dept_merge_col = []

        for d in ["last", "next"]:
            for event in ["National", "Cultural", "Religious"]:
                dept_event_data = calendar_data.query(f"{d}_{event}_distance < 7 and target >= 0")\
                                                .groupby(["dept_id", f"{d}_{event}_name"])["target"]\
                                                .agg(["mean", "std", "count"])
                dept_event_data.columns = [f"{d}_{event}_dept_" + c for c in dept_event_data.columns]
                dept_event_data = dept_event_data.reset_index().merge(dept_data.reset_index(),
                                                                            on="dept_id",
                                                                            how="left")
                dept_event_data[f"{d}_{event}_dept_mean_ratio"] = dept_event_data[f"{d}_{event}_dept_mean"] / dept_event_data["mean"]
                dept_event_data[f"{d}_{event}_dept_p_value"] = np.log10(np.vectorize(get_p)(dept_event_data[f"{d}_{event}_dept_mean"],
                                                                                            dept_event_data[f"{d}_{event}_dept_std"],
                                                                                            dept_event_data[f"{d}_{event}_dept_count"],
                                                                                            dept_event_data["mean"],
                                                                                            dept_event_data["std"],
                                                                                            dept_event_data["count"]))
                calendar_data = calendar_data.merge(dept_event_data[["dept_id", f"{d}_{event}_name", f"{d}_{event}_dept_mean_ratio", f"{d}_{event}_dept_p_value"]],
                                                    on=["dept_id", f"{d}_{event}_name"],
                                                    how="left")
                calendar_data[f"{d}_{event}_name"] = calendar_data[f"{d}_{event}_dept_mean_ratio"].astype(np.float32)
                calendar_data[f"{d}_{event}_dept_p_value"] = calendar_data[f"{d}_{event}_dept_p_value"].astype(np.float32)
                dept_merge_col.append(f"{d}_{event}_dept_mean_ratio")
                dept_merge_col.append(f"{d}_{event}_dept_p_value")
        datas.append(calendar_data[dept_merge_col])         
        
        # SNAPの情報を利用
        snap_df = self.make_snap(calendar_df)        
        datas.append(data_df[["d", "state_id"]].merge(snap_df,
                                                      on=["d", "state_id"],
                                                      how="left")
                                               .drop(["d", "state_id"], axis=1))
        tmp_df = data_df.merge(snap_df,
                               on=["d", "state_id"],
                               how="left")
        pivot = tmp_df.query("target >= 0").pivot_table(columns="snap",
                                                        index="item_id",
                                                        values="target",
                                                        aggfunc=["mean", "std", "count"])
        pivot["snap_ttest_pvalue"] = np.vectorize(get_p)(pivot[("mean", 0)], pivot[("std", 0)], pivot[("count", 0)], pivot[("mean", 1)], pivot[("std", 1)], pivot[("count", 1)])                                                       
        tmp_df = tmp_df.merge(pivot.reset_index()[["item_id", "snap_ttest_pvalue"]],
                            on="item_id",
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

        calendar_df[["wday", "month", "year", 'day', 'dayofyear']] = calendar_df[["wday", "month", "year", 'day', 'dayofyear']].astype(np.uint16)
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
