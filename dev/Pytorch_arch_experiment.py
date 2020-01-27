#!/usr/bin/env python
# coding: utf-8

#
# old code for Google Colab
#

# !pip install -U -q PyDrive

# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive
# from google.colab import auth
# from oauth2client.client import GoogleCredentials

# 1. Authenticate and create the PyDrive client.
# auth.authenticate_user()
# gauth = GoogleAuth()
# gauth.credentials = GoogleCredentials.get_application_default()
# drive = GoogleDrive(gauth)

### Download datasets
# downloaded = drive.CreateFile({'id':"1t5wi4xjrOD5OcxH3SMhTNH9wxRuU5dMb"})   # replace the id with id of file you want to access
# downloaded.GetContentFile('NKAS_Raman.hdf5')  

# downloaded = drive.CreateFile({'id':"1s62a9Tfgmht0lUCjlAwVjW56vbngWE26"})   # replace the id with id of file you want to access
# downloaded.GetContentFile('DataSet_0p20val.hdf5')  

# downloaded = drive.CreateFile({'id':"1FxLvyBgmfQ17xctfOjzEvWcNmGa9F4sR"})   # replace the id with id of file you want to access
# downloaded.GetContentFile('NKAS_density.hdf5')  

# downloaded = drive.CreateFile({'id':"1AyFwtkEzhH01clvoo9Y5_unGfeMM8Q1E"})   # replace the id with id of file you want to access
# downloaded.GetContentFile('neuravi.py') 

#
# Library loading
#


import matplotlib.pyplot as plt # plotting
import numpy as np 
import pandas as pd # manipulate dataframes

import matplotlib, time, h5py, torch

from tqdm import tqdm
from sklearn.metrics import mean_squared_error

# First we check if CUDA is available
print("CUDA AVAILABLE? ",torch.cuda.is_available())

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
      
device = get_default_device()
print(device)

import neuravi # import my stuff

np.random.seed = 167 # fix random seed for reproducibility

#
# Start calculations
#
nb_exp = 2000
nb_neurons = np.random.randint(10,high=500,size=nb_exp)
nb_layers = np.random.randint(1,high=10,size=nb_exp)
p_drop = np.around(np.random.random_sample(nb_exp)*0.5,2)

for i in tqdm(range(nb_exp)):
        
    # name for saving
    name = "./model/exp_arch/l"+str(nb_layers[i])+"_n"+str(nb_neurons[i])+"_p"+str(p_drop[i])+".pth"
    
	# custom data loader, automatically sent to device
    ds = neuravi.data_loader("./data/DataSet_0p20val.hdf5",
			     "./data/NKAS_Raman.hdf5",
                             "./data/NKAS_density.hdf5",
                             "./data/NKAS_optical.hdf5",
                             device)
    
    # declaring model
    neuralmodel = neuravi.model(4,nb_neurons[i],nb_layers[i],ds.nb_channels_raman,p_drop=p_drop[i]) 

    # criterion for match
    criterion = torch.nn.MSELoss()
    criterion.to(device) # sending criterion on device
    optimizer = torch.optim.Adam(neuralmodel.parameters(), lr = 0.001) # optimizer

    # we initialize the output bias and send the neural net on device
    neuralmodel.output_bias_init()
    neuralmodel.to(device)
    
    #
    # PRETRAINING
    #
    neuralmodel, record_pretrain_loss, record_prevalid_loss = neuravi.training(neuralmodel,ds,criterion,optimizer,name, mode="pretrain",verbose=False)
                
    #
    # TRAINING
    #
    neuralmodel, record_train_loss, record_valid_loss = neuravi.training(neuralmodel,ds,criterion,optimizer,name,train_patience=50,verbose=False)





