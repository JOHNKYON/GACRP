"""This is a data loader."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
from functools import reduce

import pandas as pd
from pandas.io.json import json_normalize


class DataBoy:
    """This class load data, pre-process data."""

    def __init__(self, data_path):
        """
        Initialization.
        :param (string) data_path: The data directory.
        """
        self.data_path = data_path

    def load_data(self, paths, mode='train', json_columns='totals'):
        """
        This function load and modify them by the mode.
        :param paths: (Tuple || List) Paths of data.
        :param mode: (string) Train or predict.
        :param json_columns: Columns names of json columns.
        :return: (pandas.DataFrame) X, Y when mode == 'train'
                                    X when mode == 'test'
        """
        assert mode == 'train' or mode == 'test'
        paths = map(lambda a: self.data_path + '/' + a, paths)

        # Apply converter to conver json format data to json object
        json_conv = {col: json.loads for col in json_columns}

        data = map(lambda a: pd.read_csv(a,
                                         dtype={'fullVisitorId': str},
                                         converters=json_conv,), paths)

        data = reduce(lambda a, b: pd.concat([a, b], axis=1), data)

        # TODO: Apply json normalization for other json columns.
        for jcol in json_columns:
            temp_df = json_normalize(data[jcol])
            temp_df.columns = [jcol + '_' + col for col in temp_df.columns]
            data = data.merge(temp_df, left_index=True, right_index=True)
            data = data.drop(columns=jcol)

        if mode == 'train':
            return data.loc[:, data.columns != 'totals_transactionRevenue'], data['totals_transactionRevenue']
        else:
            return data.loc[:, data.columns != 'totals_transactionRevenue']
