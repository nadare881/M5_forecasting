from joblib import Parallel, delayed
from random import random
import numpy as np
import pandas as pd
from tqdm import tqdm

class HierarchicalMSE():
    
    def __init__(self, origin_data, label, weight=False):
        self.group_cols = list(map(list, [("all_id",),
                           ('state_id',),
                           ('store_id',),
                           ('cat_id',),
                           ('dept_id',),
                           ('state_id', 'cat_id'),
                           ('state_id', 'dept_id'),
                           ('store_id', 'cat_id'),
                           ('store_id', 'dept_id'),
                           ('item_id',),
                           ('item_id', 'state_id'),
                           ('item_id', 'store_id')]))
        self.weight_by_level = [0.45, 0.65, 0.82, 0.58, 0.74, 0.79, 0.95, 1, 1.14, 1.58, 1.64, 1.67]
        self.df = origin_data[["item_id", "dept_id", "cat_id", "store_id", "state_id", "year", "dayofyear", "sell_price"]]
        self.df["d"] = self.df["year"].map({2011: -28, 2012: 337, 2013: 703, 2014: 1068, 2015: 1433, 2016: 1798}) + self.df["dayofyear"]
        self.df["target"] = label
        self.df["pred"] = 0
        self.df["value"] = self.df["target"] * self.df["sell_price"]
        self.df["all_id"] = 0
        self.df["rolling_value"] = self.df.groupby(['item_id', 'store_id'])["value"].transform(lambda x: x.rolling(28, min_periods=1).mean())
        self.trues = dict()
        self.weight = dict()
        self.init_pweight = dict()
        self.hess = np.zeros(self.df.shape[0])
        self.n_iter = 0
        calcstreak = np.frompyfunc(lambda x, y: x + (x|y!=0), 2, 1)
        for i, cols in tqdm(enumerate(self.group_cols)):
            tcols = tuple(cols)
            self.trues[tcols] = self.df.groupby(["d"] + cols)["target"].transform("sum")
            if weight:
                """
                agg_df = self.df.groupby(("d",) + cols).agg({"target": "sum",
                                                             "rolling_value": "sum"}).reset_index()
                agg_df["numerator"] = agg_df.groupby(cols)["target"].transform(lambda x: np.sqrt(np.maximum(calcstreak.accumulate(x>0, dtype=np.object).astype(np.float64)-1, 0)).shift(28))
                agg_df["denominator"] = agg_df.groupby(cols)["target"].transform(lambda x: np.sqrt(np.square(x-x.shift(1)).cumsum()).fillna(method="bfill").shift(28))
                agg_df["weight"] = (agg_df["rolling_value"]*agg_df["numerator"]/agg_df["denominator"]*(agg_df["denominator"]>0))
                agg_df["weight"] = (agg_df["weight"] / agg_df.groupby("d")["weight"].transform("sum") * agg_df.groupby("d")["weight"].transform("count")).fillna(0)
                self.weight[cols] = self.df.merge(agg_df,
                                                  on = ("d",) + cols,
                                                  how="left")["weight"].values
                self.init_pweight[cols] = self.df["target"] / (self.trues[cols]+1e-9)
                self.hess += self.weight[cols] / 12
                
                """
                self.weight[tcols] = np.ones(self.df.shape[0]) * self.weight_by_level[i]
                self.hess = np.ones(self.df.shape[0]) * np.mean(self.weight_by_level)
            else:
                self.weight[tcols] = np.ones(self.df.shape[0])
                self.hess = np.ones(self.df.shape[0]) * np.mean(self.weight_by_level)

            
    def iter_inputs(self, y_pred):
        for level, cols in enumerate(self.group_cols, start = 1):
            yield (level, cols, self.trues[tuple(cols)], y_pred, self.df[["d", "pred"] + list(cols)], self.weight[tuple(cols)])
    
    def partial_loss(self, inp):
        level, cols, true_agg, y_pred, pdf, weight = inp
        if level == 12:
            return (y_pred-true_agg)*weight
        else:
            pred_agg = pdf.groupby(["d",] + cols)["pred"].transform("sum")
        balance = pdf["pred"]/pred_agg
        return ((pred_agg - true_agg)*balance*weight).values
    
    def partial_loss2(self, inp):
        level, cols, true_agg, y_pred, pdf, weight = inp
        if level == 12:
            return (y_pred-true_agg)*weight

        pdf["balance_pred"] = (self.df["target"]*10 + pdf["pred"]*self.n_iter) / (10 + self.n_iter)
        group = pdf.groupby(["d"] + cols)
        pred_agg = group["pred"].transform("sum")
        balance_agg = group["balance_pred"].transform("sum")
        balance = pdf["balance_pred"] / balance_agg
        return ((pred_agg - true_agg)*balance*weight).values
    
    def calc(self, y_pred, data):
        self.df["pred"] = np.maximum(y_pred, 1e-12)
        
        if (self.n_iter < 100):
            grad = Parallel(n_jobs=12, backend="threading")([delayed(self.partial_loss2)(inp) for inp in self.iter_inputs(y_pred)])
        else:
            grad = Parallel(n_jobs=12, backend="threading")([delayed(self.partial_loss)(inp) for inp in self.iter_inputs(y_pred)])
        
        # level 12

        self.n_iter += 1
        return np.mean(grad, axis=0), self.hess