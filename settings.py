import argparse
import sys
import os
from config import *


class TrainParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.args = sys.argv[1:]

        self.parser.add_argument('--dataset', type=str, required=True)
        self.parser.add_argument('--net', type=str, required=True)
        self.parser.add_argument('--project', type=str, required=True)
        self.parser.add_argument('--name', type=str, required=True)
        self.parser.add_argument('--batch_size', type=int, default=128)
        self.parser.add_argument('--act', type=str, default='relu')

        # step-wise or epoch-wise
        self.parser.add_argument('--num_workers', default=4, type=int)
        self.parser.add_argument('--num_epoch', default=120, type=int)
        # scheduler and optimizer
        self.parser.add_argument('--lr_scheduler', default='milestones',
                                 choices=['milestones', 'exp', 'cyclic'])
        self.parser.add_argument('--optimizer', default='Adam', choices=['SGD', 'Adam', 'static'])
        self.parser.add_argument('--lr', default=1.2e-3, type=float)
        # training settings
        self.parser.add_argument('--npbar', default=True, action='store_false')

        # prune settings
        self.parser.add_argument('--amount', default=0.01, type=float)
        self.parser.add_argument('--skip', default=1, type=int)
        self.parser.add_argument('--method', default=0, type=int)
        self.parser.add_argument('--amount_setting', default=0, type=int)

    def get_args(self):
        args = self.parser.parse_known_args(self.args)[0]
        return args

