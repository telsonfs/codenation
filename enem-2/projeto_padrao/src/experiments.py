from sklearn.metrics import max_error, mean_absolute_error, explained_variance_score, mean_squared_error, r2_score
from time import time


class Experiments:

    def __init__(self, x_train, y_train, x_test, y_test):

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def models_correlations(self, model):

        model_measurement = {}

        #Save model name
        model_measurement['model'] = model

        #Save execution model train time
        t0 = time()
        model.fit(self.x_train, self.y_train)
        model_measurement['train_time'] = time() - t0

        #Save execution model test time
        t0 = time()
        pred = model.predict(self.x_test)
        model_measurement['test_time'] = time() - t0

        #Save the data from execution model
        model_measurement['score'] = model.score(self.x_test, self.y_test)
        model_measurement['explained_variance_score'] = round(explained_variance_score(self.y_test, pred), 2)
        model_measurement['max_error'] = round(max_error(self.y_test, pred), 2)
        model_measurement['mean_absolute_error'] = round(mean_absolute_error(self.y_test, pred), 2)
        model_measurement['mean_squared_error'] = round(mean_squared_error(self.y_test, pred), 2)
        model_measurement['r2_score'] = round(r2_score(self.y_test, pred), 2)

        return model, model_measurement


