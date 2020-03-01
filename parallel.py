import numpy as np
from tfrbm import BBRBM, GBRBM, RegRBM
from datetime import datetime
from joblib import Parallel, delayed

data = np.array([[1, 1, 0, 0, 0],
                 [0, 1, 1, 0, 0],
                 [0, 0, 1, 1, 0],
                 [0, 0, 0, 1, 1]], float)
    
epochs = 1600
vis = data.shape[1]
hid = data.shape[0]
        
num_sim = 10

def numericSimulation(std, x):
        print('###'+str(std)+'###'+str(x)+'###'+ datetime.now().strftime("%H:%M:%S"))    
        print('REGGaussian weight init with visible+hidden bias settings')
        regrbm_normal_whb = RegRBM(n_visible=vis, n_hidden=hid, t=0.001, lam=0.001, sample_visible=False, sigma=1, learning_rate=0.05, momentum=0, 
                                   use_tqdm=False, init_weight_scheme ='gaussian', rbmName = 'Regularized Gaussian weight init with visible', stddev_par = std)
        regrbm_normal_whb.set_biases(data)
        errs = regrbm_normal_whb.fit(data, n_epoches=epochs, batch_size=1)
        regrbm_normal_whb.export_to_csv()
        regrbm_normal_whb.print_serial_number()
    
#        print('###'+str(std)+'###'+str(x)+'###'+ datetime.now().strftime("%H:%M:%S"))    
#        print('Xavier weight init')
#        bbrbm_xavier = BBRBM(n_visible=vis, n_hidden=hid, learning_rate=0.05, momentum=0, use_tqdm=False, init_weight_scheme ='xavier', 
#                             rbmName = 'Xavier weight init', stddev_par = std)
#        errs = bbrbm_xavier.fit(data, n_epoches=epochs, batch_size=1)
#        bbrbm_xavier.export_to_csv()
#        bbrbm_xavier.print_serial_number()
#        
#        print('###'+str(std)+'###'+str(x)+'###'+ datetime.now().strftime("%H:%M:%S"))    
#        print('Gaussian weight init')
#        bbrbm_normal = BBRBM(n_visible=vis, n_hidden=hid, learning_rate=0.05, momentum=0, use_tqdm=False, init_weight_scheme ='gaussian', 
#                             rbmName = 'Gaussian weight init', stddev_par = std)
#        errs = bbrbm_normal.fit(data, n_epoches=epochs, batch_size=1)
#        bbrbm_normal.export_to_csv()
#        bbrbm_normal.print_serial_number()
#        
#        print('###'+str(std)+'###'+str(x)+'###'+ datetime.now().strftime("%H:%M:%S"))    
#        print('Perturbed REMERGE weight init')
#        bbrbm_premerge = BBRBM(n_visible=vis, n_hidden=hid, learning_rate=0.05, momentum=0, use_tqdm=False, 
#                               init_weight_scheme ='perturbed remerge', rbmName = 'Perturbed REMERGE weight init', stddev_par = std)
#        errs = bbrbm_premerge.fit(data, n_epoches=epochs, batch_size=1)
#        bbrbm_premerge.export_to_csv()
#        bbrbm_premerge.print_serial_number()
        
#MISSING 0.05 !!!!
Parallel(n_jobs=4)(delayed(numericSimulation)(std,x) for std in list(reversed([0.01, 0.1, 0.2, 0.5, 1, 2])) for x in range(num_sim))
