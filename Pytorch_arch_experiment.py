#!/usr/bin/env python
# coding: utf-8

# In[1]:


# old code for Google Colab

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


# In[2]:


# Library loading
get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd # manipulate dataframes
import matplotlib.pyplot as plt # plotting
import matplotlib
import numpy as np
import time

from sklearn.metrics import mean_squared_error
import h5py

# Check torch install
try:
  import torch
except:
  print("Starting a session, torch not installed, installing...")
  get_ipython().system('pip3 install torch # we install torch if not installed')
  import torch

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

# import my stuff
import neuravi


# In[5]:


nb_exp = 1000
nb_neurons = np.random.randint(10,high=500,size=nb_exp)
nb_layers = np.random.randint(1,high=10,size=nb_exp)
p_drop = np.around(np.random.random_sample(nb_exp)*0.5,2)

for i in range(nb_exp):
    print('Experiment {} started...'.format(i))
    
    # name for saving
    name = "./model/exp_arch/l"+str(nb_layers[i])+"_n"+str(nb_neurons[i])+"_p"+str(p_drop[i])+".pth"
    
    ds = neuravi.data_loader("./data/DataSet_0p20val.hdf5","./data/NKAS_Raman.hdf5","./data/NKAS_density.hdf5",device)
    
    # declaring model
    neuralmodel = neuravi.model(4,nb_neurons,nb_layers,ds.nb_channels_raman,p_drop=p_drop) 

    # criterion for match
    criterion = torch.nn.MSELoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(neuralmodel.parameters(), lr = 0.001) # optimizer

    # we initialize the output bias
    neuralmodel.output_bias_init()

    # we send the neural net on device
    neuralmodel.to(device)
    
    #
    # PRETRAINING
    #
    neuralmodel, record_pretrain_loss, record_prevalid_loss = neuravi.pretraining(neuralmodel,ds,criterion,optimizer,verbose=False)
                
    #
    # TRAINING
    #
    neuralmodel, record_train_loss, record_valid_loss = neuravi.maintraining(neuralmodel,ds,criterion,optimizer,name,verbose=False)


# In[ ]:




