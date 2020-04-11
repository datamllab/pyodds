from pyodds.automl.config_space import CUMULATIVE_SEARCH_SPACE, construct_classifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from hyperopt import fmin, tpe, Trials, space_eval
from sklearn.metrics import roc_auc_score, mean_squared_error


class Cash():
    def __init__(self, data, ground_truth):

        self.train, self.test, self.test_ground_truth = self.split(data,
                                                                   ground_truth)
        self.count = 0
        self.best_classifier = None

    @staticmethod
    def split(data, ground_truth=None):
        split_ratio = 0.66
        num_samples = data.shape[0]
        split = int(num_samples * split_ratio)
        if ground_truth is not None:
            return data.iloc[:split], data.iloc[split:], ground_truth[split:]
        else:
            return data.iloc[:split], data.iloc[split:], None

    def objective_function(self, param, count):
        clf = construct_classifier(param)
        clf.fit(self.train)

        predictions = clf.predict(self.test)
        anomaly_scores = clf.anomaly_likelihood(self.test)
        if self.test_ground_truth is not None:
            # plot_predictions(predictions, self.test_ground_truth,
            #                  param['type'] + str(count) + ".png")
            loss = -1 * max(roc_auc_score(self.test_ground_truth, anomaly_scores),
                            1 - roc_auc_score(self.test_ground_truth,
                                              anomaly_scores))
        else:
            # Trade-off to prefer FPs over FNs - Practical application HealthCare
            loss = mean_squared_error(predictions,
                                      np.asarray([0] * len(predictions)))
        return loss

    def f(self, params):
        self.count += 1
        return self.objective_function(params, self.count)

    def model_selector(self, max_evals=50):
        trials = Trials()
        best_clf = fmin(self.f, CUMULATIVE_SEARCH_SPACE, algo=tpe.suggest,
                        max_evals=max_evals, trials=trials)
        config = space_eval(CUMULATIVE_SEARCH_SPACE, best_clf)
        print(config)
        return construct_classifier(config)
