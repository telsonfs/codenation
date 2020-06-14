from sklearn import metrics
from time import time


class Experiments:

    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def models_classification(self, model):
        model_measurement = {'model': model}

        # Save model name

        # Save execution model train time
        t0 = time()
        model.fit(self.x_train, self.y_train)
        model_measurement['train_time'] = time() - t0

        # Save execution model test time
        t0 = time()
        predict = model.predict(self.x_test)
        model_measurement['test_time'] = time() - t0

        # Save the data from execution model
        model_measurement['score'] = metrics.accuracy_score(self.y_test, predict)
        model_measurement['confusion_matrix'] = metrics.confusion_matrix(self.y_test, predict)
        model_measurement['classification_report'] = metrics.classification_report(self.y_test, predict)

        return model, model_measurement
