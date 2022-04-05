# Import all required packages
from sklearn.base import BaseEstimator, TransformerMixin
import holidays
import pandas as pd


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
    # def __init__(self):
    #     self = self

    def fit(self, X, y=None):
        return self

    # Method that describes what we need this transformer to do
    def transform(self, X, y=None):
        return X


class ProcessDWT(BaseEstimator, TransformerMixin):
    # def __init__(self):
    #     self = self

    def fit(self, X, y=None):
        return self

    # Method that describes what we need this transformer to do
    def transform(self, X, y=None):
        return X


class ProcessVessel(BaseEstimator, TransformerMixin):
    # def __init__(self):
    #     self = self

    def fit(self, X, y=None):
        return self

    # Method that describes what we need this transformer to do
    def transform(self, X, y=None):
        return X


class DropDate(BaseEstimator, TransformerMixin):
    def __init__(self, date_cols=[]):
        self.date_cols = date_cols

    def fit(self, X, y=None):
        return self

    # Method that describes what we need this transformer to do
    def transform(self, X, y=None):
        for col in self.date_cols:
            X.drop(col, axis=1, inplace=True)
        return X

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
