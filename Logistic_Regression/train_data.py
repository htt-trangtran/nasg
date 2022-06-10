""" 
Training
"""

import numpy as np
import pandas as pd

from load_data import *
from algorithms import *
from record_history import *
from util_func import *
from schedule_LR import *

def train_data (dataname, num_epoch, namealg, listeta, listrecord, record_path):
    # Loading data and start network
    #-------------------------------------------------------------------------------
    print('Step 0: Import data')
    # Load data
    X, Y, X_test, Y_test = import_data(dataname)
    lamb =  0.01
    shuffle = 1

    #-------------------------------------------------------------------------------
    # Start training

    for eta in listeta:
        # Pick LR scheme: 
        scheduleLR = constant (eta)
        record_name = dataname + namealg + str(eta) 
        listrecord.append(record_name)

        for seed in range(10):
            record_seed = record_name + '_seed_' + str(seed)
            # Pick algorithm 
            if namealg == '_NASG_':
              NASG_train (X, Y, X_test, Y_test, num_epoch, shuffle, lamb, scheduleLR, record_path, record_seed)
            elif namealg == '_NASGPI_':
              NASGPI_train (X, Y, X_test, Y_test, num_epoch, shuffle, lamb, scheduleLR, record_path, record_seed)
            elif namealg == '_NAG_':
              NAG_train (X, Y, X_test, Y_test, num_epoch, lamb, scheduleLR, record_path, record_seed)
            elif namealg == '_SGDM_':
              SGDM_train (X, Y, X_test, Y_test, num_epoch, shuffle, lamb, scheduleLR, 0.9, record_path, record_seed)
            elif namealg == '_SGD_':
              SGD_train (X, Y, X_test, Y_test, num_epoch, shuffle, lamb, scheduleLR, record_path, record_seed)
            elif namealg == '_ADAM_':
              ADAM_train (X, Y, X_test, Y_test, num_epoch, shuffle, lamb, scheduleLR, record_path, record_seed)
          
    return listrecord