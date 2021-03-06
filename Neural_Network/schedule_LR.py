"""
Different learning rates
"""

import math 

def constant(eta):
    x = lambda t : eta 
    return x

def diminishing(eta, alpha):
    return lambda t :  eta / ((t + alpha)** (1/3))

def exponential(eta, alpha):
    return lambda t : eta * (alpha ** t)

def cosine (eta, T):
    return lambda t : eta * 0.5 * (1 + math.cos(t*math.pi/T))





