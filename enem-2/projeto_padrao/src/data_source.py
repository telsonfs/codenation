import pandas as pd 

class DataSource:

    def __init__(self):
        self.path_train = '../data/train.csv'
        self.path_test = '../data/test.csv'

    def read_data(self, treino = True):

        if etapa_treino:
            df = pd.read_csv(self.path_train)
            return df

        df = pd.read_csv(self.path_test)
        return df
