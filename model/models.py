"""This is a module of multiple models"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import lightgbm as lgb


class LightGBM:
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
        """
        Train a lightGBM model.
        :param X: (pandas.DataFrame) Training data X.
        :param Y: (pandas.DataFrame) Training data Y.
        :param X_valid: (pandas.DataFrame) Cross validation data X.
        :param Y_valid: (pandas.DataFrame) Cross validation data Y.
        :param feature_names: (pandas.DataFrame) Labels for X.
        :param categorical_features: (pandas.DataFrame) Labels for categorical fetures in X.
        :return: (LightGBM) The class itself.
        """

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
        """
        Predict output Y with the trained model and input X.
        :param X: (pandas.DataFrame) Prediction input data X.
        :param feature_names: feature_names: (pandas.DataFrame) Labels for X.
        :param categorical_features: (pandas.DataFrame) Labels for categorical fetures in X.
        :return: (pandas.DataFrame) The prediction result.
        """
        return self.model.predict(X, num_iteration=self.model.best_iteration)
