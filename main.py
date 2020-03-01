# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:27:18 2020

@author: folde
"""

import numpy as np
import matplotlib.pyplot as plt
from tfrbm import BBRBM, GBRBM, RegRBM

data = np.array([[1, 1, 0], 
              [0, 1, 1]], float)

num_sim = 5

for x in range(num_sim):
    for std in [0.1, 0.2]:

        print('###'+str(x)+'###')
        print('Xavier weight init')
        bbrbm_xavier = BBRBM(n_visible=3, n_hidden=2, learning_rate=0.05, momentum=0, use_tqdm=False, init_weight_scheme ='xavier', rbmName = 'Xavier weight init', stddev_par = std)
        # print(bbrbm_xavier.get_weights())
        errs = bbrbm_xavier.fit(data, n_epoches=400, batch_size=1)
        # plt.plot(errs)
        # plt.show()
        # print(bbrbm_xavier.get_weights())
        bbrbm_xavier.export_to_csv()
    
        # print('Uniform one weight init')
        # bbrbm_ones = BBRBM(n_visible=3, n_hidden=2, learning_rate=0.05, momentum=0, use_tqdm=False, init_weight_scheme ='ones', rbmName = 'Uniform one weight init')
        # # print(bbrbm_ones.get_weights())
        # errs = bbrbm_ones.fit(data, n_epoches=400, batch_size=1)
        # # plt.plot(errs)
        # # plt.show()
        # # print(bbrbm_ones.get_weights())
        # bbrbm_ones.export_to_csv()
    
    
        #from the Hinton guide "If the statistics used for learning are stochastic, the initial
        # weights can all be zero since the noise in the statistics will make the hidden units become different
        # from one another even if they all have identical connectivities."
        # print('Uniform zero weight init')
        # bbrbm_zeros = BBRBM(n_visible=3, n_hidden=2, learning_rate=0.05, momentum=0, use_tqdm=False, init_weight_scheme ='zeros', rbmName = 'Uniform zero weight init')
        # # print(bbrbm_zeros.get_weights())
        # errs = bbrbm_zeros.fit(data, n_epoches=400, batch_size=1)
        # # plt.plot(errs)
        # # plt.show()
        # # print(bbrbm_zeros.get_weights())
        # bbrbm_zeros.export_to_csv()
        # #according to the Hinton guide gaussian weight init with mean=0 and stddev=0.01
        
        print('Gaussian weight init')
        bbrbm_normal = BBRBM(n_visible=3, n_hidden=2, learning_rate=0.05, momentum=0, use_tqdm=False, init_weight_scheme ='gaussian', rbmName = 'Gaussian weight init', stddev_par = std)
        # print(bbrbm_normal.get_weights())
        errs = bbrbm_normal.fit(data, n_epoches=400, batch_size=1)
        # plt.plot(errs)
        # plt.show()
        # print(bbrbm_normal.get_weights())
        bbrbm_normal.export_to_csv()
    
        #according to the Hinton: gaussian weight + incorporating recommendation for visible biases
        
        # print('Gaussian weight init with visible bias settings')
        # bbrbm_normal_wb = BBRBM(n_visible=3, n_hidden=2, learning_rate=0.05, momentum=0, use_tqdm=False, init_weight_scheme ='gaussian', rbmName  = 'Gaussian weight init with visible bias settings')
        # bbrbm_normal_wb.set_biases(data)
        
        # # print(bbrbm_normal_wb.get_weights())
        # errs = bbrbm_normal_wb.fit(data, n_epoches=400, batch_size=1)
        # # plt.plot(errs)
        # # plt.show()
        # # print(bbrbm_normal_wb.get_weights())
        # bbrbm_normal_wb.export_to_csv()
    
        # print('Gaussian weight init with visible+hidden bias settings')
        # bbrbm_normal_whb = BBRBM(n_visible=3, n_hidden=2, learning_rate=0.05, momentum=0, use_tqdm=False, init_weight_scheme ='gaussian', rbmName = 'Gaussian weight init with visible bias settings')
        # bbrbm_normal_whb.set_biases(data, 0.5)
        
        # # print(bbrbm_normal_whb.get_weights())
        # errs = bbrbm_normal_whb.fit(data, n_epoches=400, batch_size=1)
        # # plt.plot(errs)
        # # plt.show()
        # # print(bbrbm_normal_whb.get_weights())
        # bbrbm_normal_whb.export_to_csv()
    
        # print('Gaussian weight init with visible+hidden bias settings+')
        # bbrbm_normal_whb = BBRBM(n_visible=3, n_hidden=2, learning_rate=0.05, momentum=0, use_tqdm=False, init_weight_scheme ='gaussian', rbmName = 'Gaussian weight init with visible+hidden bias settings+')
        # bbrbm_normal_whb.set_biases(data, 0.9)
        
        # # print(bbrbm_normal_whb.get_weights())
        # errs = bbrbm_normal_whb.fit(data, n_epoches=400, batch_size=1)
        # # plt.plot(errs)
        # # plt.show()
        # # print(bbrbm_normal_whb.get_weights())
        # bbrbm_normal_whb.export_to_csv()
    
        
        ########## REGULARIZATION
        
        print('REGGaussian weight init with visible+hidden bias settings')
        regrbm_normal_whb = RegRBM(n_visible=3, n_hidden=2, t=0.001, lam=0.001, sample_visible=False, sigma=1, learning_rate=0.05, momentum=0, use_tqdm=False, init_weight_scheme ='gaussian', rbmName = 'Regularized Gaussian weight init with visible')
        regrbm_normal_whb.set_biases(data)
        
        # print(regrbm_normal_whb.get_weights())
        errs = regrbm_normal_whb.fit(data, n_epoches=400, batch_size=1)
        # plt.plot(errs)
        # plt.show()
        # print(regrbm_normal_whb.get_weights())
        regrbm_normal_whb.export_to_csv()
    
    
        print('REMERGE weight init')
        bbrbm_remerge = BBRBM(n_visible=3, n_hidden=2, learning_rate=0.05, momentum=0, use_tqdm=False, init_weight_scheme ='remerge', rbmName = 'REMERGE weight init', stddev_par = std)
        # print(bbrbm_xavier.get_weights())
        errs = bbrbm_remerge.fit(data, n_epoches=400, batch_size=1)
        # plt.plot(errs)
        # plt.show()
        # print(bbrbm_xavier.get_weights())
        bbrbm_remerge.export_to_csv()
        
        print('Perturbed REMERGE weight init')
        bbrbm_premerge = BBRBM(n_visible=3, n_hidden=2, learning_rate=0.05, momentum=0, use_tqdm=False, init_weight_scheme ='perturbed remerge', rbmName = 'Perturbed REMERGE weight init', stddev_par = std)
        # print(bbrbm_xavier.get_weights())
        errs = bbrbm_premerge.fit(data, n_epoches=400, batch_size=1)
        # plt.plot(errs)
        # plt.show()
        # print(bbrbm_xavier.get_weights())
        bbrbm_premerge.export_to_csv()
        
#%% n_visible=vis, n_hidden=hid,
        

import numpy as np
import matplotlib.pyplot as plt
from tfrbm import BBRBM, GBRBM

data = np.array([[1, 1, 0, 0, 0],
                 [0, 1, 1, 0, 0],
                 [0, 0, 1, 1, 0],
                 [0, 0, 0, 1, 1]], float)
    
epochs = 1600
vis = data.shape[1]
hid = data.shape[0]
        
num_sim = 10

for x in range(num_sim):
    for std in [0.01, 0.05, 0.1, 0.2, 0.5, 1, 2]:
        print('###'+str(std)+'###'+str(x)+'###')
              
        print('Xavier weight init')
        bbrbm_xavier = BBRBM(n_visible=vis, n_hidden=hid, learning_rate=0.05, momentum=0, use_tqdm=False, init_weight_scheme ='xavier', 
                             rbmName = 'Xavier weight init', stddev_par = std)
        # print(bbrbm_xavier.get_weights())
        errs = bbrbm_xavier.fit(data, n_epoches=epochs, batch_size=1)
        # plt.plot(errs)
        # plt.show()
        # print(bbrbm_xavier.get_weights())
        bbrbm_xavier.export_to_csv()
    
        # print('Uniform one weight init')
        # bbrbm_ones = BBRBM(n_visible=vis, n_hidden=hid, learning_rate=0.05, momentum=0, use_tqdm=False, init_weight_scheme ='ones', rbmName = 'Uniform one weight init')
        # # print(bbrbm_ones.get_weights())
        # errs = bbrbm_ones.fit(data, n_epoches=1200, batch_size=1)
        # # plt.plot(errs)
        # # plt.show()
        # # print(bbrbm_ones.get_weights())
        # bbrbm_ones.export_to_csv()
    
    
        #from the Hinton guide "If the statistics used for learning are stochastic, the initial
        # weights can all be zero since the noise in the statistics will make the hidden units become different
        # from one another even if they all have identical connectivities."
        # print('Uniform zero weight init')
        # bbrbm_zeros = BBRBM(n_visible=vis, n_hidden=hid, learning_rate=0.05, momentum=0, use_tqdm=False, init_weight_scheme ='zeros', rbmName = 'Uniform zero weight init')
        # # print(bbrbm_zeros.get_weights())
        # errs = bbrbm_zeros.fit(data, n_epoches=1200, batch_size=1)
        # # plt.plot(errs)
        # # plt.show()
        # # print(bbrbm_zeros.get_weights())
        # bbrbm_zeros.export_to_csv()
        # #according to the Hinton guide gaussian weight init with mean=0 and stddev=0.01
        
        print('Gaussian weight init')
        bbrbm_normal = BBRBM(n_visible=vis, n_hidden=hid, learning_rate=0.05, momentum=0, use_tqdm=False, init_weight_scheme ='gaussian', 
                             rbmName = 'Gaussian weight init', stddev_par = std)
        # print(bbrbm_normal.get_weights())
        errs = bbrbm_normal.fit(data, n_epoches=epochs, batch_size=1)
        # plt.plot(errs)
        # plt.show()
        # print(bbrbm_normal.get_weights())
        bbrbm_normal.export_to_csv()
    
        #according to the Hinton: gaussian weight + incorporating recommendation for visible biases
        
        # print('Gaussian weight init with visible bias settings')
        # bbrbm_normal_wb = BBRBM(n_visible=vis, n_hidden=hid, learning_rate=0.05, momentum=0, use_tqdm=False, init_weight_scheme ='gaussian', rbmName  = 'Gaussian weight init with visible bias settings')
        # bbrbm_normal_wb.set_biases(data)
        
        # # print(bbrbm_normal_wb.get_weights())
        # errs = bbrbm_normal_wb.fit(data, n_epoches=1200, batch_size=1)
        # # plt.plot(errs)
        # # plt.show()
        # # print(bbrbm_normal_wb.get_weights())
        # bbrbm_normal_wb.export_to_csv()
    
        # print('Gaussian weight init with visible+hidden bias settings')
        # bbrbm_normal_whb = BBRBM(n_visible=vis, n_hidden=hid, learning_rate=0.05, momentum=0, use_tqdm=False, init_weight_scheme ='gaussian', rbmName = 'Gaussian weight init with visible bias settings')
        # bbrbm_normal_whb.set_biases(data, 0.5)
        
        # # print(bbrbm_normal_whb.get_weights())
        # errs = bbrbm_normal_whb.fit(data, n_epoches=1200, batch_size=1)
        # # plt.plot(errs)
        # # plt.show()
        # # print(bbrbm_normal_whb.get_weights())
        # bbrbm_normal_whb.export_to_csv()
    
        # print('Gaussian weight init with visible+hidden bias settings+')
        # bbrbm_normal_whb = BBRBM(n_visible=vis, n_hidden=hid, learning_rate=0.05, momentum=0, use_tqdm=False, init_weight_scheme ='gaussian', rbmName = 'Gaussian weight init with visible+hidden bias settings+')
        # bbrbm_normal_whb.set_biases(data, 0.9)
        
        # # print(bbrbm_normal_whb.get_weights())
        # errs = bbrbm_normal_whb.fit(data, n_epoches=1200, batch_size=1)
        # # plt.plot(errs)
        # # plt.show()
        # # print(bbrbm_normal_whb.get_weights())
        # bbrbm_normal_whb.export_to_csv()
    
        
        ########## REGULARIZATION
        
    #    print('REGGaussian weight init with visible+hidden bias settings')
    #    regrbm_normal_whb = RegRBM(n_visible=vis, n_hidden=hid, t=0.001, lam=0.001, sample_visible=False, sigma=1, learning_rate=0.05, momentum=0, use_tqdm=False, init_weight_scheme ='gaussian', rbmName = 'Regularized Gaussian weight init with visible')
    #    regrbm_normal_whb.set_biases(data)
    #    
    #    # print(regrbm_normal_whb.get_weights())
    #    errs = regrbm_normal_whb.fit(data, n_epoches=1200, batch_size=1)
    #    # plt.plot(errs)
    #    # plt.show()
    #    # print(regrbm_normal_whb.get_weights())
    #    regrbm_normal_whb.export_to_csv()
    
    
#        print('REMERGE weight init')
#        bbrbm_remerge = BBRBM(n_visible=vis, n_hidden=hid, learning_rate=0.05, momentum=0, use_tqdm=False, 
#                              init_weight_scheme ='remerge', rbmName = 'REMERGE weight init', stddev_par = std)
#        # print(bbrbm_xavier.get_weights())
#        errs = bbrbm_remerge.fit(data, n_epoches=epochs, batch_size=1)
#        # plt.plot(errs)
#        # plt.show()
#        # print(bbrbm_xavier.get_weights())
#        bbrbm_remerge.export_to_csv()
        
        print('Perturbed REMERGE weight init')
        bbrbm_premerge = BBRBM(n_visible=vis, n_hidden=hid, learning_rate=0.05, momentum=0, use_tqdm=False, 
                               init_weight_scheme ='perturbed remerge', rbmName = 'Perturbed REMERGE weight init', stddev_par = std)
        # print(bbrbm_xavier.get_weights())
        errs = bbrbm_premerge.fit(data, n_epoches=epochs, batch_size=1)
        # plt.plot(errs)
        # plt.show()
        # print(bbrbm_xavier.get_weights())
        bbrbm_premerge.export_to_csv()