#!/usr/bin/env python
# coding: utf-8
# (c) Charles Le Losq 2022
# see embedded licence file
# imelt V1.2

#
# Library Loading
#
import pandas as pd # manipulate dataframes
import matplotlib.pyplot as plt # plotting
import numpy as np
np.random.seed = 167 # fix random seed for reproducibility

import time, torch, os

# local imports
import imelt as imelt

from sklearn.metrics import mean_squared_error

from tqdm import tqdm

#
# First we check if CUDA is available
#
print("CUDA AVAILABLE? ",torch.cuda.is_available())

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device = get_default_device()
print(device)

#
# Data loading
#
# custom data loader, automatically sent to device
ds = imelt.data_loader()
ds.print_data()

#
# Training 1 model
#

# reference architecture: 6 layers, 400 neurons per layer
# Parameters were tuned after the random search, & learning rate by Bayesian Optimization & patience by hand.

nb_layers = 4
nb_neurons = 400
p_drop = 0.15

print("Network architecture is: {} layers, {} neurons/layers, dropout {}".format(nb_layers,nb_neurons,p_drop))

# Create directories if they do not exist
imelt.create_dir('./model/')
imelt.create_dir('./figures/')
imelt.create_dir('./outputs/')
imelt.create_dir('./figures/single/')
imelt.create_dir('./outputs/single/')

name = "./model/l"+str(nb_layers)+"_n"+str(nb_neurons)+"_p"+str(p_drop)+"_GELU_cpfree"+".pth"

# declaring model
neuralmodel = imelt.model(ds.x_visco_train.shape[1],
                          hidden_size=nb_neurons,num_layers=nb_layers,nb_channels_raman=ds.nb_channels_raman,
                          activation_function = torch.nn.GELU(), p_drop=p_drop)

# criterion for match
criterion = torch.nn.MSELoss(reduction='mean')
criterion.to(device) # sending criterion on device

# we initialize the output bias and send the neural net on device
neuralmodel.output_bias_init()
neuralmodel = neuralmodel.float()
neuralmodel.to(device)

#
# TRAINING
#
time1 = time.time()
optimizer = torch.optim.Adam(neuralmodel.parameters(), lr = 0.0003) # optimizer
neuralmodel, record_train_loss, record_valid_loss = imelt.training(neuralmodel,ds,
                                                                     criterion,optimizer,save_switch=True,save_name=name,
                                                                     train_patience=250,min_delta=0.05,
                                                                     verbose=True)

# The following was for use with the LBFGS optimizer
# Results are not very conclusive, not very stable compared to ADAM for this problem
# optimizer = torch.optim.LBFGS(neuralmodel.parameters(),lr=0.1) # optimizer
# neuralmodel, record_train_loss, record_valid_loss = imelt.training_lbfgs(neuralmodel,ds,
#                                                                      criterion,optimizer,save_switch=True,save_name=name,
#                                                                      train_patience=250,min_delta=0.05,
#                                                                      verbose=True)

time2 = time.time()
print("It took {:.1f} seconds".format(time2-time1))
print(name)
plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
plt.plot(record_train_loss,label="train")
plt.plot(record_valid_loss,label="valid")
plt.legend()
# with zoom after 600 epochs
plt.subplot(2,1,2)
plt.plot(record_train_loss,label="train")
plt.plot(record_valid_loss,label="valid")
plt.xlim(600,)
plt.ylim(-10,10)
plt.legend()
plt.savefig("./figures/single/loss.pdf")