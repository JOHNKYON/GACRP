"""
This is main class. It loads parameters and send them to different trainer.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from trainer.trainer import Trainer


def main():
    # TODO: Implement argparse
    params = {'n_estimators': 10000,
              'num_leaves': 30,
              'learning_rate': 0.01}

    training_params = {'early_stop_round': 100,
                       'verbose': 100}

    trainer = Trainer('lightgbm', params,training_params)
    trainer.train()


if __name__ == '__main__':
    main()
