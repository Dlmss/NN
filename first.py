# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 18:30:28 2017

@author: Ewan
"""

import matplotlib.pyplot as plt
import numpy as np
from math_util import relu, softmax, relu_backward, softmax_backward


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def initialize_parameters(layers_dims):
    """
    
    Function which initializes the different parameters of the model.
    
    @param layers_dims -- a list of the different sizes of each layers.
    
    @return parameters -- a tuple of the parameters
    
    """
    
    L = len(layers_dims)
    parameters = {}
    np.random.seed(seed)
    
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
    
    return parameters


def linear_forward(W, A, b):
    """
    
    Implement the calculus of Z
    
    @param W --
           A --
           b --
    
    @return Z -- the output of the layer
    
    """
    print("W :", W)
    print("A :", A)
    print("b :", b)
    Z = np.dot(W, A) + b
    linear_cache = (W, A, b)
    print("Z : ", Z)
    return Z, linear_cache


def linear_activation_forward(W, A_prev, b, activation):
    """
    
    Implement
    
    @param
    
    @return
    
    """
    
    Z, linear_cache = linear_forward(W, A_prev, b)
    
    if activation == "softmax":
        A = softmax(Z)
    
    if activation == "relu":
        A = relu(Z)
    
    activation_cache = Z
    caches = (linear_cache, activation_cache)
    
    return A, caches
    

def L_model_forward(X, parameters):
    """
    
    Implement
    
    @param
    
    @return
    
    """
    
    caches = []
    A = X
    L = len(parameters) // 2
    
    for l in range(1, L):
        A_prev = A
        print("Couche " + str(l))
        A, cache = linear_activation_forward(parameters['W' + str(l)], A_prev, parameters['b' + str(l)], "relu")
        caches.append(cache)
#        print("A.shape :", A.shape)
#        print("A : ", A)
#        print("cache.len :", len(cache))
#        print("cache :", cache)
    
    AL, cache = linear_activation_forward(parameters['W' + str(L)], A, parameters['b' + str(L)], "softmax")
    caches.append(cache)
    return AL, caches


def compute_cost(Y, AL):
    """
    
    Implement
    
    @param
    
    @return
    
    """
    
    m = Y.shape[0]
    
    loss = - np.sum(np.multiply(Y.T, np.log(AL)), axis=0)
    cost = (1/m) * np.sum(loss)
    
    return cost


def linear_backward(dZ, cache):
    """
    
    Implement
    
    @param
    
    @return dA --
            dW --
            db --
    
    """
    
    m = dZ.shape[1]
    W, A_prev, _ = cache
    
    dA = np.dot(W.T, dZ)
    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis = 1, keepdims = True)
   
    return dA, dW, db
    
def linear_activation_backward(dA_prev, caches, activation):
    """
    
    Implement
    
    @param
    
    @return dA --
            dW --
            db --
    
    """
    
    linear_cache, activation_cache = caches
    
    if activation == "softmax":
        dZ = softmax_backward(dA_prev, activation_cache)
        
    if activation == "relu":
        dZ = relu_backward(dA_prev, activation_cache)

    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dW, dA_prev, db
    
        
def L_model_backward(Y, AL, caches):
    """
    
    Implement
    
    @param
    
    @return grads -- a list of the gradients
    
    """
    
    grads = {}
    L = len(caches)
    current_cache = caches[L-1]
    
    dAL = AL - Y  #In reality it's dZL = AL - Y but it's to try
    grads["dW" + str(L)], grads["dA" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "softmax")
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        grads["dW" + str(l+1)], grads["dA" + str(l+1)], grads["db" + str(l+1)] = linear_activation_backward(grads["dA" + str(l+2)], current_cache, "relu")
    
    return grads


def update_parameters(grads, parameters, learning_rate = 0.01):
    """
    
    Implement
    
    @param
    
    @return grads -- a list of the gradients
    
    """
    
    L = len(parameters) // 2
    
    for l in range(L):
        parameters['W' + str(l+1)] -= learning_rate * grads["dW" + str(l+1)]
        print("W" + str(l+1) + " : ", parameters['W' + str(l+1)])
        parameters['b' + str(l+1)] -= learning_rate * grads["db" + str(l+1)]
    
    return parameters

    
def main():
    global seed
    seed = 1
    learning_rate = 0.01
    layers_dims = [3072, 14, 12, 10]
    
    for k in range(1, 2):
        dataset = unpickle('D:\Programmation\Python\cifar-10-batches-py\data_batch_' + str(k))
        X_origin = dataset[b'data']
        Y_origin = dataset[b'labels']
        X = np.transpose(X_origin)
        Y = np.transpose(Y_origin)
        costs = []
        parameters = initialize_parameters(layers_dims)
        
        for iteration in range(2):
            AL, caches = L_model_forward(X, parameters)
            cost = compute_cost(Y, AL)
            costs.append(cost)
            grads = L_model_backward(Y, AL, caches)
            parameters = update_parameters(grads, parameters, learning_rate)
            
            

main()

# TEST
#global seed
#seed = 1
#learning_rate = 0.01
#layers_dims = [6, 3, 2, 10]
#    
#x = np.array([[ 59, 154, 255, 71, 250,  62],
#              [ 43, 126, 253, 60, 254,  61],
#              [ 50, 105, 253, 74, 211,  60],
#              [140, 139,  83, 68, 215, 130],
#              [ 84, 142,  83, 69, 255, 130],
#              [ 72, 144,  84, 68, 254, 131]])
#y = np.array([6, 9, 9, 1, 1, 5])
#X = x
#Y = y
#print(X)
#costs = []
#parameters = initialize_parameters(layers_dims)
#
#for iteration in range(3):
#    AL, caches = L_model_forward(X, parameters)
#    print("AL : ", AL)
#    cost = compute_cost(Y, AL)
#    costs.append(cost)
#    grads = L_model_backward(Y, AL, caches)
#    print("grads : ", grads)
#    parameters = update_parameters(grads, parameters, learning_rate)
#    print("costs : ", costs)
