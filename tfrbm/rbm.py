from __future__ import print_function

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import pandas as pd
import sys
import math
from .util import tf_xavier_init
import csv
import datetime
import os
from random import randint

class RBM:
    def __init__(self, 
                 n_visible,
                 n_hidden,
                 learning_rate=0.01,
                 momentum=0.95,
                 xavier_const=1.0,
                 err_function='mse',
                 use_tqdm=False,
                 # DEPRECATED:
                 tqdm=None,
                 init_weight_scheme ='xavier',
                 init_bias_scheme = 'zeros',
                 rbmName = None,
                 stddev_par = 0.05):
        
        dirname = os.path.dirname(__file__)
        
        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        random_id = randint(1000,9999)
        
        self.serial_number =  self.current_time + '_' + str(random_id)

        self.train_log_dir_raw_params =  os.path.join(dirname, 'logs/rbms/raw/params/' + self.current_time + '_' + str(random_id))        
                
        # initialize list
        hyperparams = [self.current_time, rbmName, n_visible, n_hidden, learning_rate, momentum, xavier_const, err_function, init_weight_scheme, init_bias_scheme, stddev_par] 
          
        # Create the pandas DataFrame 
        self.df_hyperparams = pd.DataFrame([hyperparams], columns = ['Timestamp','rbmName', 'n_visible', 'n_hidden', 'learning_rate','momentum','xavier_const','err_function','init_weight_scheme','init_bias_scheme', 'stddev_par']) 
        
        self.train_log_dir_raw_vals =  os.path.join(dirname, 'logs/rbms/raw/vals/' + self.current_time + '_' + str(random_id))
        
        self.train_log_dir_raw_svd =  os.path.join(dirname, 'logs/rbms/raw/svd/' + self.current_time + '_' + str(random_id))
        
        self.train_log_dir_raw_weight =  os.path.join(dirname, 'logs/rbms/raw/weight/' + self.current_time + '_' + str(random_id))

        
        self.val_meanError = []
        self.val_pearson = []
        
        self.val_svd = []

        if not 0.0 <= momentum <= 1.0:
            raise ValueError('momentum should be in range [0, 1]')

        if err_function not in {'mse', 'cosine'}:
            raise ValueError('err_function should be either \'mse\' or \'cosine\'')

        self._use_tqdm = use_tqdm
        self._tqdm = None

        if use_tqdm or tqdm is not None:
            from tqdm import tqdm
            self._tqdm = tqdm

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.rbmName = rbmName
        self.init_weight_scheme = init_weight_scheme
        self.init_bias_scheme = init_bias_scheme
        
        # self.writer = tf.compat.v1.summary.FileWriter(train_log_dir)

        self.x = tf.compat.v1.placeholder(tf.float32, [None, self.n_visible])
        self.y = tf.compat.v1.placeholder(tf.float32, [None, self.n_hidden])

        if self.init_weight_scheme == 'xavier':
            self.w = tf.Variable(tf_xavier_init(self.n_visible, self.n_hidden, const=xavier_const), dtype=tf.float32)
        elif self.init_weight_scheme == 'ones':
            self.w = tf.Variable(tf.ones([self.n_visible, self.n_hidden], tf.dtypes.float32))
        elif self.init_weight_scheme == 'zeros':
            self.w = tf.Variable(tf.zeros([self.n_visible, self.n_hidden], tf.dtypes.float32))
        elif self.init_weight_scheme == 'gaussian':
            self.w = tf.Variable(tf.random.normal([self.n_visible, self.n_hidden], mean=0.0, stddev=stddev_par), dtype=tf.float32)
        elif self.init_weight_scheme == 'remerge':
            if n_visible==3 and n_hidden==2:    
                self.w = tf.Variable([[1, 0], [1, 1], [0, 1]], dtype=tf.float32)
            elif n_visible==4 and n_hidden==3:
                self.w = tf.Variable([[1, 0, 0], [1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=tf.float32)
            elif n_visible==5 and n_hidden==4:
                self.w = tf.Variable([[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=tf.float32)

        elif self.init_weight_scheme == 'perturbed remerge':
            if n_visible==3 and n_hidden==2:
                self.w = tf.Variable([[np.random.normal(1, stddev_par), np.random.normal(0, stddev_par)], 
                                       [np.random.normal(1, stddev_par), np.random.normal(1, stddev_par)], 
                                       [np.random.normal(0, stddev_par), np.random.normal(1, stddev_par)]], dtype=tf.float32)
            elif n_visible==4 and n_hidden==3:
                self.w = tf.Variable([[np.random.normal(1, stddev_par), np.random.normal(0, stddev_par),  np.random.normal(0, stddev_par)], 
                                       [np.random.normal(1, stddev_par), np.random.normal(1, stddev_par), np.random.normal(0, stddev_par)],
                                       [np.random.normal(0, stddev_par), np.random.normal(1, stddev_par), np.random.normal(1, stddev_par)], 
                                       [np.random.normal(0, stddev_par), np.random.normal(0, stddev_par), np.random.normal(1, stddev_par)]], dtype=tf.float32)
            elif n_visible==5 and n_hidden==4:
                self.w = tf.Variable([[np.random.normal(1, stddev_par), np.random.normal(0, stddev_par),  np.random.normal(0, stddev_par), np.random.normal(0, stddev_par)], 
                                       [np.random.normal(1, stddev_par), np.random.normal(1, stddev_par), np.random.normal(0, stddev_par), np.random.normal(0, stddev_par)],
                                       [np.random.normal(0, stddev_par), np.random.normal(1, stddev_par), np.random.normal(1, stddev_par), np.random.normal(0, stddev_par)], 
                                       [np.random.normal(0, stddev_par), np.random.normal(0, stddev_par), np.random.normal(1, stddev_par), np.random.normal(1, stddev_par)],
                                       [np.random.normal(0, stddev_par), np.random.normal(0, stddev_par), np.random.normal(0, stddev_par), np.random.normal(1, stddev_par)]], dtype=tf.float32)
            

        if self.init_bias_scheme == 'zeros':
            self.visible_bias = tf.Variable(tf.zeros([self.n_visible]), dtype=tf.float32)
            self.hidden_bias = tf.Variable(tf.zeros([self.n_hidden]), dtype=tf.float32)

        self.delta_w = tf.Variable(tf.zeros([self.n_visible, self.n_hidden]), dtype=tf.float32)
        self.delta_visible_bias = tf.Variable(tf.zeros([self.n_visible]), dtype=tf.float32)
        self.delta_hidden_bias = tf.Variable(tf.zeros([self.n_hidden]), dtype=tf.float32)

        self.update_weights = None
        self.update_deltas = None
        self.compute_hidden = None
        self.compute_visible = None
        self.compute_visible_from_hidden = None
        
        self._initialize_vars()

        assert self.update_weights is not None
        assert self.update_deltas is not None
        assert self.compute_hidden is not None
        assert self.compute_visible is not None
        assert self.compute_visible_from_hidden is not None

        if err_function == 'cosine':
            x1_norm = tf.nn.l2_normalize(self.x, 1)
            x2_norm = tf.nn.l2_normalize(self.compute_visible, 1)
            cos_val = tf.reduce_mean(tf.reduce_sum(tf.mul(x1_norm, x2_norm), 1))
            self.compute_err = tf.acos(cos_val) / tf.constant(np.pi)
        else:
            self.compute_err = tf.reduce_mean(tf.square(self.x - self.compute_visible))

        init = tf.compat.v1.global_variables_initializer()

        self.sess = tf.compat.v1.Session()

        self.sess.run(init)
        
    def print_serial_number(self):
        print(self.serial_number)

    def _initialize_vars(self):
        pass

    def get_err(self, batch_x):
        return self.sess.run(self.compute_err, feed_dict={self.x: batch_x})

    def get_free_energy(self):
        pass

    def transform(self, batch_x):
        return self.sess.run(self.compute_hidden, feed_dict={self.x: batch_x})

    def transform_inv(self, batch_y):
        return self.sess.run(self.compute_visible_from_hidden, feed_dict={self.y: batch_y})

    def reconstruct(self, batch_x):
        return self.sess.run(self.compute_visible, feed_dict={self.x: batch_x})

    def partial_fit(self, batch_x):
        self.sess.run(self.update_weights + self.update_deltas, feed_dict={self.x: batch_x})

    def fit(self,
            data_x,
            n_epoches=10,
            batch_size=10,
            shuffle=True,
            verbose=True):
        
        assert n_epoches > 0

        n_data = data_x.shape[0]

        if batch_size > 0:
            n_batches = n_data // batch_size + (0 if n_data % batch_size == 0 else 1)
        else:
            n_batches = 1

        if shuffle:
            data_x_cpy = data_x.copy()
            inds = np.arange(n_data)
        else:
            data_x_cpy = data_x

        errs = []

        for e in range(n_epoches):
            #if verbose and not self._use_tqdm:
                #print('Epoch: {:d}'.format(e))

            epoch_errs = np.zeros((n_batches,))
            epoch_errs_ptr = 0

            if shuffle:
                np.random.shuffle(inds)
                data_x_cpy = data_x_cpy[inds]

            r_batches = range(n_batches)

            if verbose and self._use_tqdm:
                r_batches = self._tqdm(r_batches, desc='Epoch: {:d}'.format(e), ascii=True, file=sys.stdout)

            for b in r_batches:
                batch_x = data_x_cpy[b * batch_size:(b + 1) * batch_size]
                self.partial_fit(batch_x)
                
                
                batch_err = self.get_err(batch_x)
                epoch_errs[epoch_errs_ptr] = batch_err
                epoch_errs_ptr += 1

            # summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag='mean err',
            #                                          simple_value=epoch_errs.mean())])
            
            self.val_meanError.append(epoch_errs.mean())
            
            # self.writer.add_summary(summary, e)            
            
            if self.n_visible==3 and self.n_hidden==2:
                pearson = np.corrcoef(np.concatenate(self.sess.run(self.w)),np.concatenate(np.array([[1, .0], [1, 1], [.0, 1]])))[0,1]
            elif self.n_visible==4 and self.n_hidden==3:
                pearson = np.corrcoef(np.concatenate(self.sess.run(self.w)),np.concatenate(np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1], [0, 0, 1]])))[0,1]
            elif self.n_visible==5 and self.n_hidden==4:
                pearson = np.corrcoef(np.concatenate(self.sess.run(self.w)),np.concatenate(np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]])))[0,1]

            # summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag='pearson',
            #                                          simple_value=pearson)])
            
            self.val_pearson.append(pearson)
                                     
            try:
                s, u, v = tf.linalg.svd(self.w)
                self.val_svd.append({'Timestamp': self.current_time,'Epoch': e, 's': self.sess.run(s), 'u': self.sess.run(u), 'v': self.sess.run(v)})
            except:
                pass                     

            if verbose:
                err_mean = epoch_errs.mean()
                if self._use_tqdm:
                    self._tqdm.write('Train error: {:.4f}'.format(err_mean))
                    self._tqdm.write('')
                #else:
                    #print('Train error: {:.4f}'.format(err_mean))
                    #print('')
                sys.stdout.flush()

            errs = np.hstack([errs, epoch_errs])
            
#            self.log_histogram('W', self.w, e)

        return errs

    def get_weights(self):
        return self.sess.run(self.w),\
            self.sess.run(self.visible_bias),\
            self.sess.run(self.hidden_bias)
            

    def save_weights(self, filename, name):
        saver = tf.train.Saver({name + '_w': self.w,
                                name + '_v': self.visible_bias,
                                name + '_h': self.hidden_bias})
        return saver.save(self.sess, filename)

    def set_weights(self, w, visible_bias, hidden_bias):
        self.sess.run(self.w.assign(w))
        self.sess.run(self.visible_bias.assign(visible_bias))
        self.sess.run(self.hidden_bias.assign(hidden_bias))

    def load_weights(self, filename, name):
        saver = tf.train.Saver({name + '_w': self.w,
                                name + '_v': self.visible_bias,
                                name + '_h': self.hidden_bias})
        saver.restore(self.sess, filename)
        
    def set_biases(self, data, target_probability=0):
        
        data_mean = np.mean(data,0)
        
        def f(x):
            return math.log(x/(1-x+10**(-5)))
        
        data_logprob = np.array([f(xi) for xi in data_mean])
        self.visible_bias = tf.convert_to_tensor(data_logprob, tf.dtypes.float32)
        
        if target_probability != 0:
            target = np.repeat(target_probability, data.shape[1])
            data_targetprob = np.array([f(xi) for xi in target])
            self.hidden_bias =  tf.convert_to_tensor(data_targetprob, tf.dtypes.float32)
            
    def log_histogram(self, tag, values, step, bins=1000):
        """Logs the histogram of a list/vector of values."""
        # Convert to a numpy array
        values = self.sess.run(self.w)
        
        
        
        
        # Create histogram using numpy        
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.compat.v1.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        # summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, histo=hist)])
        # self.writer.add_summary(summary, step)
        # self.writer.flush()
        
    def export_to_csv(self):
        # Export as csv
        self.df_hyperparams.to_csv(self.train_log_dir_raw_params+'.csv', index = False)
        df_meanError = pd.DataFrame({'Timestamp':[self.current_time], 'Val':['meanError'],'num':[self.val_meanError]})
        df_pearson = pd.DataFrame({'Timestamp':[self.current_time], 'Val':['pearson'],'num':[self.val_pearson]})
        df_weight = pd.DataFrame({'Timestamp':[self.current_time], 'Val':['weight'], 'num':[self.sess.run(self.w)]})
        df_vbias = pd.DataFrame({'Timestamp':[self.current_time], 'Val':['vbias'], 'num':[self.sess.run(self.visible_bias)]})
        df_hbias = pd.DataFrame({'Timestamp':[self.current_time], 'Val':['hbias'], 'num':[self.sess.run(self.hidden_bias)]})

        csv_file = self.train_log_dir_raw_svd + '_svd.csv'
        csv_columns = ['Timestamp','Epoch','s','u','v']
        
        try:
            with open(csv_file, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                writer.writeheader()
                for data in self.val_svd:
                    writer.writerow(data)
        except IOError:
            print("I/O error")
        
        df_meanError.to_csv(self.train_log_dir_raw_vals+'_meanError.csv', index=False)
        df_pearson.to_csv(self.train_log_dir_raw_vals+'_pearson.csv', index=False)
        df_weight.to_csv(self.train_log_dir_raw_weight+'_weight.csv', index=False)
        df_vbias.to_csv(self.train_log_dir_raw_weight+'_vbias.csv', index=False)
        df_hbias.to_csv(self.train_log_dir_raw_weight+'_hbias.csv', index=False)
        
        
        
        
    
        
