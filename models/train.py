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

def load_datasets(feats):
    dfs = [pd.read_pickle(f'../processed/{f}_data.ftr') for f in feats]
    return pd.concat(dfs, axis=1)

modelname = "hoge"
LAG = 14

if __name__ == "__main__":
    data_df = pd.read_pickle("../processed/base_data.pickle")
    
    feats = ["Base", "Calendar", "Price", f"Rolling_id_LAG_{LAG}", "LastYear_stat", f"Prophet_level_oof_LAG_{LAG}", "Prophet_item_oof", "Prophet_id_oof"]
    data_df = pd.concat([data_df, load_datasets(feats)], axis=1)

    drop_col = ["target", "d", "id", "next_change", "rchange_rate"]

    test_data_df = data_df.query(f"{1913} < d and d <= {1913+28}")
    val_data_df = data_df.query(f"{1913-28} < d and d <= {1913}")
    val_data_df["target"] = val_data_df["target"].fillna(0)
    val_data_df2 = data_df.query(f"{1913} < d and d <= {1941}")
    val_data_df2["target"] = val_data_df2["target"].fillna(0)

    year = 6
    dev_data = lgb.Dataset(data_df.query(f"{1913 - 365*year} <= d and d <= 1885").drop(drop_col, axis=1), label=data_df.query(f"{1913 - 365*year} <= d and d <= {1913}")["target"])#, weight=data_df.query("d <= 1885")["weight"])
    val_data = lgb.Dataset(val_data_df.drop(drop_col, axis=1), label=val_data_df["target"])#, weight=val_data_df["weight"])
    val_data_2 = lgb.Dataset(val_data_df2.drop(drop_col, axis=1), label=val_data_df2["target"])
    val_data.duration = "val"
    val_data_2.duration = "test"

    hierarchicalMSE = HierarchicalMSE(data_df.query(f"{1913 - 365*year} <= d and d <= {1913} and target>=0"), data_df.query(f"{1913 - 365*year} <= d and d <= {1913} and target>=0")["target"].values, True)

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
    clf = lgb.train(param, dev_data, num_round, valid_sets = [val_data, val_data_2], verbose_eval=1, early_stopping_rounds = 200, feval=evaluator.feval, fobj=hierarchicalMSE.calc, evals_result=history)
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
    val_res.to_csv(f"{modelname}_val.csv", index=None)

    test_res = val_data_df2[["id", "target", "d"]]
    test_res["id"] = test_res["id"].map(idrmap)
    test_res["cat_id"] = test_res["id"].map(lambda x: "_".join(x.split("_")[0:1]))
    test_res["dept_id"] = test_res["id"].map(lambda x: "_".join(x.split("_")[0:2]))
    test_res["item_id"] = test_res["id"].map(lambda x: "_".join(x.split("_")[0:3]))
    test_res["state_id"] = test_res["id"].map(lambda x: "_".join(x.split("_")[3:4]))
    test_res["store_id"] = test_res["id"].map(lambda x: "_".join(x.split("_")[3:5]))
    test_res["pred"] = clf.predict(test_data_df.drop(drop_col, axis=1), num_iteration=clf.best_iteration)
    test_res.to_csv(f"{modelname}_test.csv", index=None)

    pd.DataFrame(evaluator_val.history).to_csv(f"{modelname}_history_spec_val.csv", index=None)
    pd.DataFrame(evaluator_test.history).to_csv(f"{modelname}_history_spec_test.csv", index=None)
