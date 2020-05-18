import os
import numpy as np

from utils import prepare_env


class Model:
    def __init__(self, metadata):
        root_path, models_path = prepare_env()

        self.metadata = metadata
        self.done_training = False

        from sol.meta_model import MetaModel
        self.model = MetaModel(metadata['class_num'], models_path, root_path)
        print(metadata)

    def train(self, train_dataset, remaining_time_budget=None):
        X, y = train_dataset
        if remaining_time_budget < 60:
            self.done_training = True

        r = self.model.train_for_budget(X, y)
        if r:
            self.done_training = True

    def test(self, X, remaining_time_budget=None):
        results = np.zeros(
            (self.metadata['test_num'], self.metadata['class_num'])
        )
        y_pred = self.model.predict(X)

        results[:y_pred.shape[0], :y_pred.shape[1]] = y_pred
        return results
