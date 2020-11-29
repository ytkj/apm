from typing import List, Dict

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna.integration.lightgbm as olgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GroupKFold, KFold

from pandas_utils import DF, ND, S, is_pd


class BaseTransformer(BaseEstimator, TransformerMixin):

    def fit(self, _: DF, __=None):
        return self

    def transform(self, x: DF) -> DF:
        return x


class ColumnTransformer:

    def __init__(self, defs: Dict[str, BaseTransformer]):
        self.defs = defs

    def fit(self, x: DF, y=None):
        for col, transformer in self.defs.items():
            transformer.fit(x[col], y)
        return self

    def transform(self, x: DF) -> DF:
        xp = x.copy()
        for col, transformer in self.defs.items():
            xp[col] = transformer.transform(x[col])
        return xp

    def fit_transform(self, x: DF, y=None) -> DF:
        xp = x.copy()
        for col, transformer in self.defs.items():
            if hasattr(transformer, 'fit_transform'):
                xp[col] = transformer.fit_transform(x[col], y)
            else:
                xp[col] = transformer.fit(x[col], y).transform(x[col])
        return xp


class BinaryTransformer(BaseTransformer):

    def transform(self, s_in: S) -> S:
        s = s_in.copy()
        s[:] = np.where(s > 0, 1, 0)
        return s


class BaseEnsembleCV(BaseEstimator):

    def __init__(self, n_splits: int = 5, seed: int = 31):
        self.n_splits = n_splits
        self.seed = seed
        self.models = None
        self.y_oof = None

    def fit_(self, x_train: DF, y_train: S, x_val: DF, y_val: S, i_fold: int) -> BaseEstimator:
        raise NotImplementedError()

    def predict_(self, model: BaseEstimator, x: DF) -> ND:
        raise NotImplementedError()

    def fit(self, x: DF, y: S, groups=None):
        self.models = []
        self.y_oof = np.zeros(y.shape)
        if is_pd(x):
            self.y_oof = pd.Series(self.y_oof)
            
        if groups is None:
            kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed).split(x, y)
        else:
            kfold = GroupKFold(n_splits=self.n_splits).split(x, y, groups=groups)
            
        for i_fold, (i_train, i_val) in enumerate(kfold):
            if is_pd(x):
                x_train, x_val = x.loc[i_train, :], x.loc[i_val, :]
            else:
                x_train, x_val = x[i_train], x[i_val]
            y_train, y_val = y[i_train], y[i_val]
            model = self.fit_(x_train, y_train, x_val, y_val, i_fold)
            self.models.append(model)
            self.y_oof[i_val] = self.predict_(model, x_val)
        
        return self
    

class OptunaLgbECV(BaseEnsembleCV):
    
    def __init__(self, lgb_params: dict, **kwargs):
        super().__init__(**kwargs)
        self.lgb_params = lgb_params
        self.best_params = None
    
    def fit_(self, x_train: DF, y_train: S, x_val: DF, y_val: S, i_fold: int) -> lgb.Booster:
        d_train = lgb.Dataset(x_train, label=y_train)
        d_val = lgb.Dataset(x_val, label=y_val)
        
        if self.best_params is None:            
            model = olgb.train(
                params=self.lgb_params,
                train_set=d_train,
                valid_sets=(d_val, d_train),
                valid_names=('val', 'train'),
                verbose_eval=10,
            )
            self.best_params = model.params
        else:
            model = lgb.train(
                params=self.best_params,
                train_set=d_train,
                valid_sets=(d_val, d_train),
                valid_names=('val', 'train'),
                verbose_eval=10,
            )
        return model
    
    def predict_(self, model: lgb.Booster, x: DF) -> ND:
        return model.predict(x)
