""" 
Training
"""

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

def train_data (dataname, num_epoch, namealg, listeta, listrecord, record_path):
    # Loading data and start network
    #-------------------------------------------------------------------------------
    print('Step 0: Load data and start network')
    # Load data
    train_loader, test_loader = load_data (dataname, 256)

    # Start network
    if (dataname == 'cifar10'):
      net = Linear_cifar()
      name_net = Linear_cifar   
    if (dataname == 'fashionmnist') or (dataname == 'mnist'):
      net = Linear_mnist()
      name_net = Linear_mnist
    print('---', name_net, 'started')

    # GPU :)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Loss Function
    criterion = nn.CrossEntropyLoss()

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
              NASG_train (name_net, num_epoch, train_loader, test_loader, scheduleLR, criterion, record_path, record_seed)
            elif namealg == '_SGDM_':
              SGDM_train (name_net, num_epoch, train_loader, test_loader, scheduleLR, 0.9, criterion, record_path, record_seed)
            elif namealg == '_SGD_':
              SGD_train (name_net, num_epoch, train_loader, test_loader, scheduleLR, criterion, record_path, record_seed)
            elif namealg == '_ADAM_':
              ADAM_train (name_net, num_epoch, train_loader, test_loader, scheduleLR, criterion, record_path, record_seed)
          
    return listrecord