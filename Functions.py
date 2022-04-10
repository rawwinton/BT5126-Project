# Import all required packages
from sklearn.base import BaseEstimator, TransformerMixin
import holidays
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder



class ProcessDates(BaseEstimator, TransformerMixin):
    # def __init__(self):
    #     self = self

    def fit(self, X, y=None):
        return self

    # Method that describes what we need this transformer to do
    def transform(self, X, y=None):
        br_holidays = holidays.Brazil()
        X['year'] = X.Date.apply(lambda x: int(x.strftime('%Y')))
        X['month_num'] = X.Date.apply(lambda x: int(x.strftime('%m')))
        # X['month'] = X.Date.apply(lambda x: x.strftime('%b'))
        X['day_of_month'] = X.Date.apply(lambda x: int(x.strftime('%d')))
        X['day_of_year'] = X.Date.apply(lambda x: int(x.strftime('%j')))
        X['week_of_year'] = X.Date.apply(lambda x: int(x.strftime('%W')))
        X['is_weekend'] = X.Date.apply(lambda x: int(x.strftime('%w')) > 5)

        # X = pd.merge(left=X, right=pd.get_dummies(X['month'], prefix='month_'), left_index=True, right_index=True, )

        return X


class ProcessHolidays(BaseEstimator, TransformerMixin):
    # def __init__(self):
    #     self = self

    def fit(self, X, y=None):
        return self

    # Method that describes what we need this transformer to do
    def transform(self, X, y=None):
        br_holidays = holidays.Brazil()
        X['is_holiday'] = X.Date.apply(lambda x: x.strftime('%Y-%m-%d') in br_holidays)

        # X = pd.merge(left=X, right=pd.get_dummies(X['month'], prefix='month_'), left_index=True, right_index=True, )

        return X


class ProcessWeather(BaseEstimator, TransformerMixin):
    def __init__(self, action='actual'):
        self.action = action

    def fit(self, X, y=None):
        return self

    # Method that describes what we need this transformer to do
    def transform(self, X, y=None):
        if self.action == 'drop':
            return X.drop(['tempt', 'prcp'], axis=1, inplace=False)
        else:
            return X


class ProcessWSpd(BaseEstimator, TransformerMixin):
    def __init__(self, action='actual'):
        self.action = action

    def fit(self, X, y=None):
        return self

    # Method that describes what we need this transformer to do
    def transform(self, X, y=None):
        if self.action == 'drop':
            return X.drop(['wspd'], axis=1, inplace=False)
        elif self.action == 'actual':
            return X
        else:
            X['wspd'] = X['wspd'].apply(lambda x: self.get_wind_cat(x))
            ohe = OneHotEncoder(categories=[['Fresh_Breeze', 'Strong_Breeze', 'Near_Gale', 'Others']],
                                handle_unknown='ignore', sparse=False)
            sic = pd.DataFrame(ohe.fit_transform(X[['wspd']]), columns=ohe.get_feature_names_out(), index=X.index)
            return X.drop(columns=['wspd']).join(sic)

    @staticmethod
    def get_wind_cat(x):
        # Ref: https://www.windfinder.com/wind/windspeed.htm
        if 21 >= x:
            x = 'Fresh_Breeze'
        elif x <= 27:
            x = 'Strong_Breeze'
        elif x <= 33:
            x = 'Near_Gale'
        else:
            x = 'Others'
        return x


class ProcessVessel(BaseEstimator, TransformerMixin):
    def __init__(self, vessel_cols=[], dwt_cols=[], usage='both'):
        self.vessel_cols = vessel_cols
        self.dwt_cols = dwt_cols
        self.usage = usage

    def fit(self, X, y=None):
        return self

    # Method that describes what we need this transformer to do
    def transform(self, X, y=None):
        if self.usage == 'both':
            X = X
        elif self.usage == 'dwt':
            X = X.drop(self.vessel_cols[1:], axis=1, inplace=False)
        elif self.usage == 'vessel':
            X = X.drop(self.dwt_cols[1:], axis=1, inplace=False)
        return X


class DropCol(BaseEstimator, TransformerMixin):
    def __init__(self, cols=[]):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    # Method that describes what we need this transformer to do
    def transform(self, X, y=None):
        for col in self.cols:
            X.drop(col, axis=1, inplace=True)
        return X


class CreateDummyCol(BaseEstimator, TransformerMixin):
    # def __init__(self, cols=[]):
    #     self.cols = cols

    def fit(self, X, y=None):
        return self

    # Method that describes what we need this transformer to do
    def transform(self, X, y=None):
        X['dummy'] = 'XX'
        X['dummy2'] = 'XX'
        return X


class DropZeroCol(BaseEstimator, TransformerMixin):
    # def __init__(self, cols=[]):
    #     self.cols = cols

    def fit(self, X, y=None):
        return self

    # Method that describes what we need this transformer to do
    def transform(self, X, y=None):
        return X[:, ~np.all(X == 'XX', axis=0)]


# class ProcesWeather(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         self = self
#
#     def fit(self, X, y=None):
#         return self
#
#     # Method that describes what we need this transformer to do
#     def transform(self, X, y=None):
#         return X
