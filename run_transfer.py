import os
import sys
import argparse

from bisim_transfer.bisimulation import *

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument(
        '--transfer',
        default='lax',
        type=str
    )
    argparser.add_argument(
        '--src-env',
        default='FourSmallRooms_11',
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
        '-th',
        '--threshold',
        default=0.01,
        type=float
    )
    argparser.add_argument(
        '-dfk',
        '--discount-kd',
        default=0.9,
        type=float
    )
    argparser.add_argument(
        '-dfr',
        '--discount-r',
        default=0.1,
        type=float
    )
    argparser.add_argument(
        '--policy-dir',
        default='saved_qvalues/optimal_qvalues/',
        type=str
    )
    argparser.add_argument(
        '-l',
        '--log-dir',
        default='logs/',
        type=str
    )
    argparser.add_argument(
        '--save-dir',
        default='saved_qvalues/transferred_qvalues/',
        type=str
    )
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')

    args = argparser.parse_args()

    if args.transfer == 'basic':
        bisimulation = LaxBisimulation(args)
    elif args.transfer == 'lax':
        bisimulation = LaxBisimulation(args)
    elif args.transfer == 'pess':
        bisimulation = PessBisimulation(args)
    elif args.transfer == 'optimistic':
        bisimulation = OptBisimulation(args)
    else:
        raise ValueError("Provide a valid transfer metric")
    
    bisimulation.execute_transfer()

    bisimulation.render()
    
    print ("Transfer Accuracy: ", bisimulation.accuracy)

        
    