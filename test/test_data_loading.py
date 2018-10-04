"""This is a unit test for data loading."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.data import DataBoy


class TestDataBoy:
    """This is a class, contains multiple functions that do unit test on DataBoy class"""

    def setup(self):
        """Init DataBoy class"""
        self.databoy = DataBoy("test")

    def test_init(self):
        """Unit test for initiate DataBoy class"""
        assert self.databoy.data_path == "test"

    def test_load_data(self):
        """Unit test for data loading"""
        ids, train_x, train_y = self.databoy.load_data(['train.csv'], mode='train', json_columns=['totals'])
        assert ids.shape == (1000,)
        assert train_x.shape == (1000, 13)
        assert train_y.shape == (1000,)

        ids, test_x = self.databoy.load_data(['test.csv'], mode='test', json_columns=['totals'])
        assert ids.shape == (1000,)
        assert test_x.shape == (1000, 13)
