# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:26:28 2020

@author: folde
"""

from sklearn.metrics import mean_squared_error 
from scipy import spatial

# Given values 
Y_true = [1,1,0] # Y_true = Y (original values) 

# calculated values 
Y_pred = [0,1,1] # Y_pred = Y' 

# Calculation of Mean Squared Error (MSE) 
mean_squared_error(Y_true,Y_pred) 
1 - spatial.distance.cosine(Y_true, Y_pred)