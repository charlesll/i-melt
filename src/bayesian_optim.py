import pandas as pd # manipulate dataframes
import matplotlib
import matplotlib.pyplot as plt # plotting
import numpy as np

import time, h5py, imelt, torch

from sklearn.metrics import mean_squared_error

from tqdm import tqdm 

from ax.plot.contour import plot_contour
from ax.plot.slice import plot_slice
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render
from ax.utils.tutorials.cnn_utils import train, evaluate

# Fixing random seeds for reproducibility
np.random.seed = 167 # fix random seed for reproducibility

#
# Training functions
#
def net_train(net, dataset, parameterization, dtype, device):
    # handling the network
    net.output_bias_init() # we initialize the output bias
    net.to(dtype=dtype, device=device) # set dtype and send to device

    # Define loss and optimizer
    criterion = torch.nn.MSELoss() # the criterion : MSE
    criterion.to(device)
    
    optimizer = torch.optim.Adam(net.parameters(), 
                                 lr=parameterization.get("lr", 0.0004), # 0.001 is used if no lr is specified
                                 weight_decay=parameterization.get("weight_decay", 0.00)) # L2 loss
                
    # training
    net, record_train_loss, record_valid_loss = imelt.training(net,dataset,criterion,optimizer,
                                                                         save_switch=False,
                                                                         nb_folds=parameterization.get("nb_folds", 1),
                                                                         train_patience=parameterization.get("patience", 150),
                                                                         min_delta=parameterization.get("min_delta", 0.05),
                                                                         verbose=False)

    # to avoid any problem with CUDA memory...
    # del neuralmodel, criterion
    # torch.cuda.empty_cache()
    
    return net, record_train_loss, record_valid_loss

def init_net(parameterization, dataset):

    # get hyperparameters for the network 
    nb_neurons = parameterization.get("nb_neurons", 200)
    nb_layers = parameterization.get("nb_layers", 4)
    p_drop = parameterization.get("p_drop", 0.2)
    
    # model declaration
    model = imelt.model(dataset.x_visco_train.shape[1],nb_neurons,nb_layers,dataset.nb_channels_raman, 
                        p_drop, activation_function = torch.nn.GELU()) # declaring model
                                  
    return model # return untrained model

def train_evaluate(parameterization):

    # Get neural net
    untrained_net = init_net(parameterization, dataset) 
    
    # train
    trained_net, record_train_loss, record_valid_loss = net_train(untrained_net, dataset, 
                            parameterization, dtype=dtype, device=device)
    
    output = {"loss_train": (np.mean(record_train_loss[-20:]), np.std(record_train_loss[-20:])), 
              "loss_valid": (np.mean(record_valid_loss[-20:]), np.std(record_valid_loss[-20:]))}
    
    # return the accuracy of the model as it was trained in this run
    return output

#
# BO experiment
#

#torch.cuda.set_device(0) #this is sometimes necessary for me
dtype = torch.float
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Calculation will be performed on {}".format(device))

# custom data loader, automatically sent to device
dataset = imelt.data_loader()

best_parameters, values, experiment, model = optimize(
    parameters=[
        #{"name": "nb_folds", "type": "range", "bounds": [5, 20]}, #20 FOLDS SEEMS GOOD
        {"name": "lr", "type": "range", "bounds": [1.0e-6, 0.001]}, #20 FOLDS SEEMS GOOD
        {"name": "nb_neurons", "type": "range", "bounds": [50, 500]}, 
        {"name": "nb_layers", "type": "range", "bounds": [2, 6]},
        {"name": "p_drop", "type": "range", "bounds": [0.1, 0.5]},
        #{"name": "min_delta", "type": "range", "bounds": [0.01, 1.0]}, # best 0.60
        #{"name": "patience", "type": "range", "bounds": [20, 500]}, # best 81
        #{"name": "weight_decay", "type": "range", "bounds": [1.0e-5, 1.0e-2]},        
    ],
    minimize=True,
    total_trials = 200,
    evaluation_function=train_evaluate,
    objective_name=('loss_valid'),
)

print(best_parameters)
means, covariances = values
print(means)
print(covariances)
#view raw
