"""This s a trainer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.model_selection import KFold
import numpy as np

from utils.data import DataBoy
from utils.submission import submission
from model.models import LightGBM


class Trainer:
    """This class contains functions that run
    the whole training and prediction process"""

    def __init__(self, model, model_params, training_params, data_path='auto', data='auto'):
        """
        Initialization.
        :param model: (str) Model name.
        :param data:  (str) data path.
        """
        self.model_name = model
        self.model_params = model_params
        self.training_params = training_params
        if data_path == 'auto':
            self.data_path = 'data'
        if data == 'auto':
            self.train_data = ['train.csv']
            self.test_data = ['test.csv']

    def train(self):
        """
        Train a model and use it to do prediction.
        :return:
        """
        # Data prepare.
        databoy = DataBoy(self.data_path)

        print("Training data loading...")
        ids, X_train, Y_train = databoy.load_data(self.train_data, mode='train')
        print("Prediction data loading...")
        ids_pre, X_pred = databoy.load_data(self.test_data, mode='test')
        Y_pred = np.zeros(X_pred.shape[0])

        kf = KFold(n_splits=5, shuffle=True)

        # Model initialization.
        if self.model_name == 'lightgbm':
            model = LightGBM(self.model_params, training_params=self.training_params)

        print("Training start.")
        for dev_index, val_index in kf.split(X_train):
            X_dev, X_val = X_train.iloc[dev_index], X_train.iloc[val_index]
            Y_dev, Y_val = Y_train.iloc[dev_index], Y_train.iloc[val_index]
            best_iteration = model.fit(X=X_dev,
                                       Y=Y_dev,
                                       X_valid=X_val,
                                       Y_valid=Y_val)

            Y_pred_temp = model.predict(X=X_pred, num_iteration=best_iteration)
            Y_pred_temp[Y_pred_temp < 0] = 0
            Y_pred = Y_pred_temp + Y_pred

        submission(ids_pre, Y_pred, 'test.csv')
