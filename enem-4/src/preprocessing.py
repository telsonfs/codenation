import pandas as pd
import tensorflow as tf
import imblearn
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE



class Preprocessing:
    def __init__(self):
        self.feature_names = None
        self.std_scaler = None
        self.catb = None
        self.scaler = None
        self.train_features = None
        self.numeric_features = None
        self.categoric_features = None

    @staticmethod
    def data_info(df):
        # for column in df.columns.to_list():
            
        #     if df[column].isnull().sum() > df.shape[0] * 0.2:
        #         df.drop(column, axis = 1, inplace = True)
        
        df.fillna(0, inplace=True)
        
        info = {'shape': df.shape, 'describe': df.describe(), 'info': df.info()}
        return info

    @staticmethod
    def select_features(df, target, numeric_features):

        x = df[numeric_features]
        # x.drop(target, axis=1, inplace = True)

        y = df[target]
        
        selector = SelectKBest(f_classif, k=8)
        selector.fit_transform(x, y)

        features = x.columns[selector.get_support(indices=True)].to_list()

        df_train = df[features]
        df_train[target] = df[target]

        return df_train

    @staticmethod
    def balancing_target(df, target):

        smote = SMOTE(sampling_strategy="minority")

        x_smote, y_smote = smote.fit_resample(df, target)

        return x_smote, y_smote








