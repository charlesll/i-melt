########## Calling relevant libraries ##########
import numpy as np
from sklearn.metrics import mean_squared_error
import scipy

import tensorflow as tf

import pandas as pd
import rampy as rp

import sklearn
import sklearn.model_selection as model_selection
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingRegressor

from sklearn.externals import joblib

#
# Data imput function
#

class Data():
    """load the data and store them in an object
    """

    def __init__(self, path="./data/"):
        self.X_train= joblib.load(path+"X_train.pkl")
        self.X_valid =joblib.load(path+"X_valid.pkl")
        self.X_test = joblib.load(path+"X_test.pkl")

        self.X_train_sc =joblib.load(path+"X_train_sc.pkl")
        self.X_valid_sc = joblib.load(path+"X_valid_sc.pkl")
        self.X_test_sc = joblib.load(path+"X_test_sc.pkl")

        self.y_train = joblib.load(path+"y_train.pkl")
        self.y_valid = joblib.load(path+"y_valid.pkl")
        self.y_test = joblib.load(path+"y_test.pkl")

        self.y_train_sc = joblib.load(path+"y_train_sc.pkl")
        self.y_valid_sc = joblib.load(path+"y_valid_sc.pkl")
        self.y_test_sc = joblib.load(path+"y_test_sc.pkl")

        self.X_scaler = joblib.load(path+"X_scaler.pkl")
        self.y_scaler = joblib.load(path+"y_scaler.pkl")

#
# TensorFlow model
#

class Model(object):
    """
    derived from  https://github.com/adventuresinML/adventures-in-ml-code/blob/master/weight_init_tensorflow.py
    """
    def __init__(self, input_size, 
                num_layers=3,
                hidden_size=100,
                dropout = 0.1,
                stdev = 0.01,
                learning_rate = 1e-4):
        
        self._input_size = input_size
        self._num_layers = num_layers # num layers does not include the input layer
        self._hidden_size = hidden_size # all layers have the same number of hidden units
        self._out_size = 5 # fixed regarding the problem shape
        self._init_bias = np.array([-4.5,np.log(40000.),np.log(4.1),np.log(10.),np.log(0.0001)])
        self._dropout = dropout
        self._stdev = stdev
        self._learning_rate = learning_rate
        self._model_def()
        
    def _model_def(self):
        # create placeholder variables
        self.input_c = tf.placeholder(dtype=tf.float32, shape=[None,self._input_size], name="chimie")
        self.input_T = tf.placeholder(dtype=tf.float32, shape=[None,1], name="T")
        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None,1], name="viscosity")

        if self._num_layers >= 1:
            
            self.Wh1 = tf.Variable(tf.random_normal([self._input_size,self._hidden_size], stddev=self._stdev, dtype=tf.float32))
            self.bh1 = tf.Variable(tf.random_normal([1,self._hidden_size], stddev=self._stdev, dtype=tf.float32))
            
            self.hidden_layer_1 = tf.nn.dropout(tf.nn.relu(tf.matmul(self.input_c, self.Wh1) + self.bh1),
                                                keep_prob=1-self._dropout,
                                                name="layer1")
            
        if self._num_layers >= 2:
            
            self.Wh2 = tf.Variable(tf.random_normal([self._hidden_size,self._hidden_size], stddev=self._stdev, dtype=tf.float32))
            self.bh2 = tf.Variable(tf.random_normal([1,self._hidden_size], stddev=self._stdev, dtype=tf.float32))

            self.hidden_layer_2 = tf.nn.dropout(tf.nn.relu(tf.matmul(self.hidden_layer_1, self.Wh2) + self.bh2),
                                                keep_prob=1-self._dropout,
                                                name="layer2")
        
        if self._num_layers >= 3:
        
            self.Wh3 = tf.Variable(tf.random_normal([self._hidden_size,self._hidden_size], stddev=self._stdev, dtype=tf.float32))
            self.bh3 = tf.Variable(tf.random_normal([1,self._hidden_size], stddev=self._stdev, dtype=tf.float32))
            
            self.hidden_layer_3 = tf.nn.dropout(tf.nn.relu(tf.matmul(self.hidden_layer_2, self.Wh3) + self.bh3),
                                                keep_prob=1-self._dropout,
                                                name="layer3")
            
        if self._num_layers >= 4:
        
            self.Wh4 = tf.Variable(tf.random_normal([self._hidden_size,self._hidden_size], stddev=self._stdev, dtype=tf.float32))
            self.bh4 = tf.Variable(tf.random_normal([1,self._hidden_size], stddev=self._stdev, dtype=tf.float32))
            
            self.hidden_layer_4 = tf.nn.dropout(tf.nn.relu(tf.matmul(self.hidden_layer_3, self.Wh4) + self.bh4),
                                                keep_prob=1-self._dropout,
                                                name="layer4")
            
        if self._num_layers == 5:
        
            self.Wh5 = tf.Variable(tf.random_normal([self._hidden_size,self._hidden_size], stddev=self._stdev, dtype=tf.float32))
            self.bh5 = tf.Variable(tf.random_normal([1,self._hidden_size], stddev=self._stdev, dtype=tf.float32))
            
            self.hidden_layer_5 = tf.nn.dropout(tf.nn.relu(tf.matmul(self.hidden_layer_4, self.Wh5) + self.bh5),
                                                keep_prob=1-self._dropout,
                                                name="layer5")

        self.Wo = tf.Variable(tf.random_normal([self._hidden_size,self._out_size], mean=0.,stddev=self._stdev, dtype=tf.float32))
        self.bo = tf.Variable(tf.random_normal([1,self._out_size], mean=self._init_bias, stddev=self._init_bias*self._stdev, dtype=tf.float32))

        if self._num_layers == 1:
            self.output = tf.add(tf.matmul(self.hidden_layer_1,self.Wo),self.bo,name="out_neurons")
        elif self._num_layers == 2:
            self.output = tf.add(tf.matmul(self.hidden_layer_2,self.Wo),self.bo,name="out_neurons")
        elif self._num_layers == 3:
            self.output = tf.add(tf.matmul(self.hidden_layer_3,self.Wo),self.bo,name="out_neurons")
        elif self._num_layers == 4:
            self.output = tf.add(tf.matmul(self.hidden_layer_4,self.Wo),self.bo,name="out_neurons")
        elif self._num_layers == 5:
            self.output = tf.add(tf.matmul(self.hidden_layer_5,self.Wo),self.bo,name="out_neurons")
        else:
            print("Error: choose between 1 to 5 layers")
        
        #
        # Adam and Gibbs with network outputs
        #
        self.ae = tf.placeholder(dtype=tf.float32, shape=[None,1], name="Ae_ph")
        self.be = tf.placeholder(dtype=tf.float32, shape=[None,1], name="Be_ph")
        self.sctg = tf.placeholder(dtype=tf.float32, shape=[None,1], name="ScTg_ph")
        self.ap = tf.placeholder(dtype=tf.float32, shape=[None,1], name="ap_ph")
        self.b = tf.placeholder(dtype=tf.float32, shape=[None,1], name="b_ph")
        
        self.ae, self.be, self.sctg, self.ap, self.b = tf.split(self.output,5,axis=1)

        # cannot be negative
        self.be = tf.exp(self.be,name="Be")
        self.sctg = tf.exp(self.sctg,name="entropy_Tg")
        self.ap = tf.exp(self.ap, name="ap")
        self.b = tf.exp(self.b, name="b")
        
        self.tg = tf.divide(self.be,np.multiply((12.0-self.ae),self.sctg),name="Tg")
        
        self.dCp = tf.add(tf.multiply(self.ap,(tf.log(self.input_T)-tf.log(self.tg))),
                          tf.multiply(self.b,(self.input_T-self.tg)),name="dCp")

        self.entropy = tf.add(self.sctg,self.dCp,name="entropy_T")

        self.denom_n =tf.multiply(self.input_T,self.entropy,name="denominator_AG")
        
        self.visco_pred = tf.add(self.ae,np.divide(self.be,self.denom_n),name="visco_pred")
        
        self.loss = tf.nn.l2_loss(self.visco_pred-self.input_y)
                                                                            
        # add the loss to the summary
        tf.summary.scalar('loss', self.loss)
        self.optimizer = tf.train.RMSPropOptimizer(self._learning_rate).minimize(self.loss)
        self.merged = tf.summary.merge_all()
        self.init_op = tf.global_variables_initializer()
