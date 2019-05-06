import os
import sys
import argparse

from bisim_transfer.bisimulation import Bisimulation

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)


    argparser.add_argument(
        '--src-env',
        default=None,
        type=str
    )
    argparser.add_argument(
        '--tgt-env',
        default=None,
        type=str
    )
    argparser.add_argument(
        '--solver',
        default='pyemd',
        type=str
    )
    argparser.add_argument(
        '--lfp-iters',
        default=5,
        type=int
    )
    argparser.add_argument(
        '-f',
        '--folder',
        type=str
    )
    argparser.add_argument(
        '-e',
        '--exp',
        type=str
    )
    argparser.add_argument(
        '-vd',
        '--val_datasets',
        dest='validation_datasets',
        nargs='+',
        default=[]
    )
    argparser.add_argument(
        '--no-train',
        dest='is_training',
        action='store_false'
    )
    argparser.add_argument(
        '-de',
        '--drive_envs',
        dest='driving_environments',
        nargs='+',
        default=[]
    )
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')

    args = argparser.parse_args()