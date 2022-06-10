"""
Run the experiments - DEMO
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

# Experiment DEMO: Comparing NASG with Other Methods
listrecord = []
num_epoch = [5, 1] # Run for only 3 epochs, and measure the performance after 1 epoch

# Data: w8a --------------------------------------------------------------------
dataname = 'w8a'

namealg = '_NASG_'
listeta = [0.01]
listrecord = train_data (dataname, num_epoch, namealg, listeta, listrecord, record_path)

namealg = '_NASGPI_'
listeta = [0.001]
listrecord = train_data (dataname, num_epoch, namealg, listeta, listrecord, record_path)

namealg = '_NAG_'
listeta = [10]
listrecord = train_data (dataname, num_epoch, namealg, listeta, listrecord, record_path)

namealg = '_SGD_'
listeta = [0.05]
listrecord = train_data (dataname, num_epoch, namealg, listeta, listrecord, record_path)

namealg = '_SGDM_'
listeta = [0.01]
listrecord = train_data (dataname, num_epoch, namealg, listeta, listrecord, record_path)

namealg = '_ADAM_'
listeta = [0.001]
listrecord = train_data (dataname, num_epoch, namealg, listeta, listrecord, record_path)

plot_data (dataname, num_epoch, listrecord, record_path, record_avg_path)