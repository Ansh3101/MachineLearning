import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer



class preProcessing:
    
    def __init__(self, df=None):
        super().__init__()
        self.df = df
    
    def dtype_select(self, dtypes=None):
        
        _train = self.df.select_dtypes(include=dtypes).columns.to_list()
        return _train
    
    
    def impute(self, numeric_cols=None, strategy='mean', missing_values=np.nan):
        
        imputer = SimpleImputer( strategy=strategy, missing_values=missing_values)
        imputer.fit(self.df[numeric_cols])
        self.df[numeric_cols] = imputer.transform(self.df[numeric_cols])
        return self.df[numeric_cols]
    
    
    def minMax(self, numeric_cols=None):
        
        scaler = MinMaxScaler()
        scaler.fit(self.df[numeric_cols])
        self.df[numeric_cols] = scaler.transform(self.df[numeric_cols])
        return self.df[numeric_cols]
    
    
    def one_hot(self, cat_cols=None):
        
        enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
        enc.fit(self.df[cat_cols])
        encoded_cols = enc.get_feature_names(cat_cols)
        self.df[encoded_cols] = enc.transform(self.df[cat_cols])
