"""
Run the experiments
"""

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import pandas as pd

from load_data import *
from algorithms import *
from record_history import *
from Lenet import *
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


# Experiment 1: Comparing NASG with Other Methods -------------------------------

num_epoch = [200, 10] # Run for 200 epochs, and measure the performance each 10 epochs

# Data: MNIST -------------------------------------------------------------------
dataname = 'mnist' 
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

# Data: FashionMNIST -----------------------------------------------------------
dataname = 'fashionmnist' 
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

# Data: Cifar10 ----------------------------------------------------------------
dataname = 'cifar10' 
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

