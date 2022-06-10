"""
Define Neural Networks 
"""

import torch.nn as nn
import torch.nn.functional as F

#-------------------------(Linear net for Cifar10)------------------------------
class Linear_cifar (nn.Module):
    def __init__ (self):
        super().__init__()

        # This network is for images of size 32x32, with 3 color channels  (Cifar10 dataset)
        self.fc1 = nn.Linear (3* 32* 32, 10)
        self.lastbias = 'fc1.bias'
    
    def forward (self, x):
        x = x.view(-1, 3* 32* 32)
        x = self.fc1(x)
        return x
#-------------------------------------------------------------------------------


#---------------(Linear net for (Fashion) MNIST)--------------------------------
class Linear_mnist (nn.Module):
    def __init__ (self):
        super().__init__()

        # This network is for images of size 28x28, with 1 color channels 
        self.fc1 = nn.Linear (28* 28, 10)
        self.lastbias = 'fc1.bias'
    
    def forward (self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        return x
#-------------------------------------------------------------------------------