import os
import pickle
import numpy as np
from torch_geometric.data import Data
import torch


def loadfolddata(datasetname, foldnum=0, base_dir='data/'):
    fold_str = str(foldnum)
    cc_path = os.path.join(base_dir, datasetname + '_5fold', 'fold' + fold_str)
    train_file_path = os.path.join(cc_path, '_x_train.pkl')
    test_file_path = os.path.join(cc_path, '_x_test.pkl')
    with open(train_file_path, 'rb') as f:
        trainlist = pickle.load(f)
    with open(test_file_path, 'rb') as ftest:
        testlist = pickle.load(ftest)
    return trainlist, testlist


def loadNewBiData(x_train, x_test):
    return x_train, x_test
