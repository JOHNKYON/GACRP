"""This is a module of multiple models"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import lightgbm as lgb


class LightGBM():
    # TODO: Log
    """
    LightBGM model class.
    This class contains functions to define a lightGBM model, train
    it and use it to predict output.
    """

    def __init__(self, params, training_params):
        """
        Initialization of a LightGBM model.
        :param params: Parameters to define a lightGBM model.
        """
        self.params = params
        self.num_boost_round = training_params['num_boost_round']
        self.early_stop_round = training_params['early_stop_round']
        self.evaluation_function = None

    def fit(self,
            X,
            Y,
            X_valid,
            Y_valid,
            feature_names='auto',
            categorical_features='auto'):

        data_train = lgb.Dataset(data=X,
                                 label=Y,
                                 feature_name=feature_names,
                                 categorical_feature=categorical_features)
        data_valid = lgb.Dataset(data=X_valid,
                                 label=Y_valid,
                                 feature_name=feature_names,
                                 categorical_feature=categorical_features)

        self.model = lgb.train(self.params,
                               data_train,
                               num_boost_round=self.num_boost_round,
                               valid_sets=data_valid,
                               early_stopping_rounds=self.early_stop_round)

        return self

    def predict(self,
                X,
                feature_names='auto',
                categorical_features='auto'):
        return self.model.predict(X, num_iteration=self.model.best_iteration)
