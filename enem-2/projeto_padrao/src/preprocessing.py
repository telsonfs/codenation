import pandas as pd

class Preprocessing:
    def __init__(self):
        self.feature_names = None
        self.std_scaler = None
        self.catb = None
        self.scaler = None
        self.train_features = None
        self.numeric_features = None
        self.categoric_features = None
    
    def data_info(self, df):
        info = {}

        info['shape'] = df.shape
        info['describe'] = df.describe()
        info['info'] = df.info()
        info['types'] = df.dtypes

        return info

    def select_features(self, df):
        return df[['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_REDACAO', 'NU_NOTA_MT']]
        


    
    
