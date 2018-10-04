"""This is utils to help submit"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np


def submission(test_ids, pred_test, file_name):
    """
    This function create submission file.
    :param test_ids: (pandas.DataFrame) fullVisitorId.
    :param pred_test: (numpy.ndarray) Prediction test.
    :param file_name: (string) Output file name.
    :return:
    """
    pred_test[pred_test < 0] = 0

    val_pred_df = pd.DataFrame(data={'fullVisitorId': test_ids,
                                     'predictedRevenue': pred_test})

    val_pred_df = val_pred_df.groupby('fullVisitorId').sum().reset_index()

    val_pred_df.columns = ['fullVIsitorId', 'predictedLogRevenue']
    val_pred_df['predictedLogRevenue'] = val_pred_df['predictedLogRevenue']
    val_pred_df.to_csv('submission/'+file_name, index=False)
