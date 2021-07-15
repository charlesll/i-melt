#!/usr/bin/env python
# coding: utf-8

#
# Load libraries
#
import pandas as pd # manipulate dataframes
import matplotlib
import matplotlib.pyplot as plt # plotting
import numpy as np

import time, h5py, imelt, torch, os

from sklearn.metrics import mean_squared_error

from tqdm import tqdm 

# First we check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Running on {} ".format(device))

# Fixing random seeds for reproducibility
# np.random.seed = 167 # fix random seed for reproducibility

#
# USEFUL FUNCTIONS
#


def train_model(ds, nb_neurons, nb_layers, p_drop, save_name, device, patience=200, min_delta=0.05):
    """function for practical training of several models"""
    neuralmodel = imelt.model(4,nb_neurons,nb_layers,ds.nb_channels_raman,p_drop=p_drop) # declaring model

    optimizer = torch.optim.Adam(neuralmodel.parameters(), lr = 0.0006) # optimizer

    # the criterion : MSE
    criterion = torch.nn.MSELoss(reduction='mean') # criterion for match, sent on device
    criterion.to(device)

    neuralmodel.output_bias_init() # we initialize the output bias
    neuralmodel = neuralmodel.float() # the model also is in Float
    neuralmodel.to(device) # we send the neural net on device
    
    # training
    neuralmodel, record_train_loss, record_valid_loss = imelt.training(neuralmodel,ds,criterion,optimizer,
                                                                         save_switch=True,save_name=save_name,
                                                                         train_patience=patience,min_delta=min_delta,verbose=False)

    # to avoid any problem with CUDA memory...
    del neuralmodel, criterion
    torch.cuda.empty_cache()

# search and create, if necessary, the folder for saving models
# Create directory
dirName = './model/exp_arch/'
try:
    # Create target Directory
    os.mkdir(dirName)
    print("Directory " , dirName ,  " Created ")
except FileExistsError:
    print("Directory " , dirName ,  " already exists")
    
    
#
# Start calculations
#
nb_exp = 3000
nb_neurons = np.random.randint(10,high=500,size=nb_exp)
nb_layers = np.random.randint(1,high=7,size=nb_exp)
p_drop = np.around(np.random.random_sample(nb_exp)*0.5,2)

# custom data loader, automatically sent to device
ds = imelt.data_loader("./data/NKAS_viscosity_reference.hdf5",
                         "./data/NKAS_Raman.hdf5",
                         "./data/NKAS_density.hdf5",
                         "./data/NKAS_optical.hdf5",
                         device)
    
for i in tqdm(range(nb_exp)):
        
    # name for saving
    name = "./model/exp_arch/l"+str(nb_layers[i])+"_n"+str(nb_neurons[i])+"_p"+str(p_drop[i])+".pth"
    
    train_model(ds,nb_neurons[i],nb_layers[i],p_drop[i], name, device)

