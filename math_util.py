# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 19:50:07 2017

@author: Ewan
"""

import numpy as np

def relu(Z):
    """
    
    Implement the ReLU function
    
    @param Z  -- matrix of the output of the linear part
    
    @return A -- the result of the ReLU function applied on Z
    
    """
    
    A = np.maximum(0, Z)
    return A


def softmax(Z):
    """
    
    Implement
    
    @param
    
    @return
    
    """
    Z_exp = np.exp(np.log(Z))
    A = Z_exp / Z_exp.sum(axis=0)
    return A


def relu_backward(dA, cache):
    """
    
    Implement the back propagation of the relu function for one layer
    dZ = dA when Z > 0 else dZ = 0
    
    @param dA    --
           cache --
    
    @return dZ   --
    
    """
    
    Z = cache
    dZ = dA
    
    dZ[Z <= 0] = 0
    
    return dZ
    

def softmax_backward(dA, cache):
    """
    
    Implement the back propagation of the softmax function for one layer
    
    @param dA    --
           cache --
    
    @return dZ   --
    
    """
    
    dZ = dA
    
    return dZ