# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import


import argparse
import train

parser = argparse.ArgumentParser(description='Binary Neural Networks')
parser.add_argument('--binary', type=str, default='connect',
        help='connect, bb or nn')
parser.add_argument('--cuda', type=bool, default=False,
        help='Use cuda or not')
parser.add_argument('--in_features', type=int, default=784,
        help='input features dim')
parser.add_argument('--out_features', type=int, default=10,
        help='output features dim')
parser.add_argument('--batch_size', type=int, default=100,
        help='batch size')
parser.add_argument('--test_batch_size', type=int, default=1000,
        help='batch size')
parser.add_argument('--lr', type=float, default=0.001,
        help='Learning rate')
parser.add_argument('--epochs', type=int, default=20,
        help='Epochs')
parser.add_argument('--optimizer', type=str, default='Adam',
        help='Adam or LFBGS w/wo LS')
parser.add_argument('--line_search_fn', type=str, default=None,
        help='None or strong_wolfe')
parser.add_argument('--filename', type=str, default='results',
        help='filename')
args = parser.parse_args()

# Redirect the output to file
import sys
sys.stdout = open('./results/' + args.filename + '.txt', 'w')

print("Arguments:", args)

train.train(args)
