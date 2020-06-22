import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
from gc import collect
import sys
from tqdm import tqdm

import warnings
warnings.simplefilter('ignore')

from objective import HierarchicalMSE
from metric import WRMSSEEvaluator, WRMSSEForLightGBM, Evaluator

modelname = "hoge"
LAG = 28
train_end = 1913

def load_datasets(feats):
    dfs = [pd.read_pickle(f'../processed/{f}_data.ftr') for f in feats]
    return pd.concat(dfs, axis=1)

if __name__ == "__main__":
    data_df = pd.read_pickle("../processed/base_data.pickle")
    
    feats = ["Base", "Calendar", "Price", f"Rolling_id_LAG_{LAG}", "LastYear_stat", f"Prophet_level_oof_LAG_{LAG}", "Prophet_item_oof", "Prophet_id_oof"]
    data_df = pd.concat([data_df, load_datasets(feats)], axis=1)

    drop_col = ["target", "d", "id", "next_change", "rchange_rate"]

    val_data_df = data_df.query(f"{1913} < d and d <= {1941}")
    val_data_df["target"] = val_data_df["target"].fillna(0)

    year = 6
    dev_data = lgb.Dataset(data_df.query(f"{train_end - 365*year} <= d and d <= {train_end}").drop(drop_col, axis=1), label=data_df.query(f"{train_end - 365*year} <= d and d <= {train_end}")["target"])
    val_data = lgb.Dataset(val_data_df.drop(drop_col, axis=1), label=val_data_df["target"])
    dev_index = data_df[["d"]].query(f"{train_end - 365*year} <= d and d <= {train_end}").index
    fake_val_index = np.random.choice(dev_index, 1000000)
    fake_val_df = data_df.iloc[fake_val_index]
    fake_val_data = lgb.Dataset(fake_val_df.drop(drop_col, axis=1), label=fake_val_df["target"])

    val_data.duration = "test"
    fake_val_data.duration = "fake"

    hierarchicalMSE = HierarchicalMSE(data_df.query(f"{train_end - 365*year} <= d and d <= {train_end} and target>=0"), data_df.query(f"{train_end - 365*year} <= d and d <= {train_end} and target>=0")["target"].values, True)

    raw_train_df = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')
    calendar = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")
    prices = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv")
    train_fold_df = raw_train_df.iloc[:, :-28]
    valid_fold_df = raw_train_df.iloc[:, -28:]
    evaluator_val = WRMSSEForLightGBM(train_fold_df, valid_fold_df, calendar, prices)

    raw_train_df = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_evaluation.csv')
    calendar = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")
    prices = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv")
    train_fold_df2 = raw_train_df.iloc[:, :-28]
    valid_fold_df2 = raw_train_df.iloc[:, -28:]
    evaluator_test = WRMSSEForLightGBM(train_fold_df2, valid_fold_df2, calendar, prices)

    evaluator = Evaluator(evaluator_val, evaluator_test)

    del data_df
    collect()

    param = {'num_leaves': 512,
            "objective": "regression",
            "metric":"None",
            'max_depth': -1,
            'learning_rate': .05,
            "boosting": "gbdt",
            "feature_fraction": 0.75,
            "bagging_freq": 1,
            "bagging_fraction": 0.9 ,
            "bagging_seed": 2434,
            "nthread":6,
            "device": "cpu",
            "verbosity": -1}

    num_round = 10000
    history = {}
    clf = lgb.train(param, dev_data, num_round, valid_sets = [val_data, fake_val_data], verbose_eval=1, early_stopping_rounds = 200, feval=evaluator.feval, fobj=hierarchicalMSE.calc, evals_result=history)
    clf.save_model(f"{modelname}.model")
    pd.DataFrame(history).to_csv(f"{modelname}_history.csv", index=None)

    raw_train_res = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_evaluation.csv')
    idmap = {v:i for i, v in enumerate(raw_train_res["id"])}
    idrmap = {v:k for k, v in idmap.items()}

    val_res = val_data_df[["id", "target", "d"]]
    val_res["id"] = val_res["id"].map(idrmap)
    val_res["cat_id"] = val_res["id"].map(lambda x: "_".join(x.split("_")[0:1]))
    val_res["dept_id"] = val_res["id"].map(lambda x: "_".join(x.split("_")[0:2]))
    val_res["item_id"] = val_res["id"].map(lambda x: "_".join(x.split("_")[0:3]))
    val_res["state_id"] = val_res["id"].map(lambda x: "_".join(x.split("_")[3:4]))
    val_res["store_id"] = val_res["id"].map(lambda x: "_".join(x.split("_")[3:5]))
    val_res["pred"] = clf.predict(val_data_df.drop(drop_col, axis=1), num_iteration=clf.best_iteration)
    val_res.to_csv(f"{modelname}_test.csv", index=None)

    pd.DataFrame(evaluator.test.history).to_csv(f"{modelname}_history_spec_test.csv", index=None)