"""
Run the experiments
"""

import os
import numpy as np
import pandas as pd

from load_data import *
from algorithms import *
from record_history import *
from util_func import *
from schedule_LR import *
from train_data import *
from average_and_plot import *

# Change the record path 
record_path = './Record/'
record_avg_path = record_path + 'Avg/'

if not os.path.exists(record_path):
    os.makedirs(record_path)

if not os.path.exists(record_avg_path):
    os.makedirs(record_avg_path)


# Experiment 1: Comparing NASG with NASG-PI and NAG ----------------------------

num_epoch = [100, 5] # Run for 100 epochs, and measure the performance each 5 epochs

# Data: w8a --------------------------------------------------------------------
dataname = 'w8a'
listrecord = []

namealg = '_NASG_'
listeta = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
listrecord = train_data (dataname, num_epoch, namealg, listeta, listrecord, record_path)

namealg = '_NASGPI_'
listeta = [10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
listrecord = train_data (dataname, num_epoch, namealg, listeta, listrecord, record_path)

namealg = '_NAG_'
listeta = [50, 10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
listrecord = train_data (dataname, num_epoch, namealg, listeta, listrecord, record_path)

plot_data (dataname, num_epoch, listrecord, record_path, record_avg_path)

# Data: ijcnn1 -----------------------------------------------------------------
dataname = 'ijcnn1'
listrecord = []

namealg = '_NASG_'
listeta = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
listrecord = train_data (dataname, num_epoch, namealg, listeta, listrecord, record_path)

namealg = '_NASGPI_'
listeta = [10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
listrecord = train_data (dataname, num_epoch, namealg, listeta, listrecord, record_path)

namealg = '_NAG_'
listeta = [50, 10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
listrecord = train_data (dataname, num_epoch, namealg, listeta, listrecord, record_path)

plot_data (dataname, num_epoch, listrecord, record_path, record_avg_path)


# Experiment 2: Comparing NASG with SGD, SGDM, ADAM ----------------------------

num_epoch = [100, 5] # Run for 100 epochs, and measure the performance each 5 epochs

# Data: w8a --------------------------------------------------------------------
dataname = 'w8a'
listrecord = []

namealg = '_NASG_'
listeta = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
listrecord = train_data (dataname, num_epoch, namealg, listeta, listrecord, record_path)

namealg = '_SGD_'
listeta = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
listrecord = train_data (dataname, num_epoch, namealg, listeta, listrecord, record_path)

namealg = '_SGDM_'
listeta = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
listrecord = train_data (dataname, num_epoch, namealg, listeta, listrecord, record_path)

namealg = '_ADAM_'
listeta = [0.005, 0.001, 0.0005]
listrecord = train_data (dataname, num_epoch, namealg, listeta, listrecord, record_path)

plot_data (dataname, num_epoch, listrecord, record_path, record_avg_path)

# Data: ijcnn1 -----------------------------------------------------------------
dataname = 'ijcnn1'
listrecord = []

namealg = '_NASG_'
listeta = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
listrecord = train_data (dataname, num_epoch, namealg, listeta, listrecord, record_path)

namealg = '_SGD_'
listeta = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
listrecord = train_data (dataname, num_epoch, namealg, listeta, listrecord, record_path)

namealg = '_SGDM_'
listeta = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
listrecord = train_data (dataname, num_epoch, namealg, listeta, listrecord, record_path)

namealg = '_ADAM_'
listeta = [0.005, 0.001, 0.0005]
listrecord = train_data (dataname, num_epoch, namealg, listeta, listrecord, record_path)

plot_data (dataname, num_epoch, listrecord, record_path, record_avg_path)

# Data: covtype -----------------------------------------------------------------
dataname = 'covtype'
listrecord = []

namealg = '_NASG_'
listeta = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
listrecord = train_data (dataname, num_epoch, namealg, listeta, listrecord, record_path)

namealg = '_SGD_'
listeta = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
listrecord = train_data (dataname, num_epoch, namealg, listeta, listrecord, record_path)

namealg = '_SGDM_'
listeta = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
listrecord = train_data (dataname, num_epoch, namealg, listeta, listrecord, record_path)

namealg = '_ADAM_'
listeta = [0.005, 0.001, 0.0005]
listrecord = train_data (dataname, num_epoch, namealg, listeta, listrecord, record_path)

plot_data (dataname, num_epoch, listrecord, record_path, record_avg_path)