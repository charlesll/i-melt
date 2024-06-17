#!/usr/bin/env python
# coding: utf-8

#
# Library loading
#

import pandas as pd # manipulate dataframes
import matplotlib
import matplotlib.pyplot as plt # plotting
import numpy as np
np.random.seed = 167 # fix random seed for reproducibility
import time, h5py, torch, shutil

from sklearn.metrics import mean_squared_error

import imelt.model as model
import imelt.utils as utils

from tqdm import tqdm

# First we check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_candidates(nb_layers = 4, nb_neurons = 300, p_drop = 0.15, nb_exp = 50):
    """ train candidates with a reference architecture"""

    print("Calculation will be performed on {}".format(device))
    
    # custom data loader, automatically sent to device
    ds = model.data_loader()

    for i in range(nb_exp):

        print("Training model {}".format(i))
        print("...\n")
        name = "./model/candidates/l"+str(nb_layers)+"_n"+str(nb_neurons)+"_p"+str(p_drop)+"_m"+str(i)+".pth"

        # declaring model
        neuralmodel = model.model(ds.x_visco_train.shape[1],nb_neurons,nb_layers,ds.nb_channels_raman,
                                  activation_function = torch.nn.GELU(), p_drop=p_drop
                                 )

        # criterion for match
        criterion = torch.nn.MSELoss(reduction='mean')
        criterion.to(device) # sending criterion on device

        # we initialize the output bias and send the neural net on device
        neuralmodel.output_bias_init()
        neuralmodel = neuralmodel.float()
        neuralmodel.to(device)

        optimizer = torch.optim.Adam(neuralmodel.parameters(), lr = 0.0003, weight_decay=0.00) # optimizer
        neuralmodel, record_train_loss, record_valid_loss = model.training(neuralmodel, ds, criterion, optimizer, 
                                                                           save_switch=True, save_name=name, nb_folds=1, train_patience=400, min_delta=0.05, verbose=True)

def detect_save(nb_layers = 4, nb_neurons = 300, p_drop = 0.15, nb_exp = 50):
    """Detect and save the best models
    
    Must run the train_candidates() function before.
    
    To select models, we use the global "loss_valid" = loss_viscosity + loss_raman + loss_density + loss_refractiveindex
    """
    
    # custom data loader, automatically sent to device
    ds = model.data_loader()
    
    # scaling coefficients for loss function
    # viscosity is always one
    # scaling coefficients for loss function
    # viscosity is always one
    ls = model.loss_scales()
    entro_scale = ls.entro
    raman_scale = ls.raman
    density_scale = ls.density
    ri_scale = ls.ri
    tg_scale = ls.tg
    
    # criterion for match
    criterion = torch.nn.MSELoss(reduction='mean')
        
    # dataframe for recording losses

    record_loss = pd.DataFrame()

    record_loss["name"] = np.zeros(nb_exp)

    record_loss["nb_layers"] = np.zeros(nb_exp)
    record_loss["nb_neurons"] = np.zeros(nb_exp)

    record_loss["loss_ag_train"] = np.zeros(nb_exp)
    record_loss["loss_ag_valid"] = np.zeros(nb_exp)

    record_loss["loss_am_train"] = np.zeros(nb_exp)
    record_loss["loss_am_valid"] = np.zeros(nb_exp)

    record_loss["loss_myega_train"] = np.zeros(nb_exp)
    record_loss["loss_myega_valid"] = np.zeros(nb_exp)

    record_loss["loss_cg_train"] = np.zeros(nb_exp)
    record_loss["loss_cg_valid"] = np.zeros(nb_exp)

    record_loss["loss_tvf_train"] = np.zeros(nb_exp)
    record_loss["loss_tvf_valid"] = np.zeros(nb_exp)

    record_loss["loss_Sconf_train"] = np.zeros(nb_exp)
    record_loss["loss_Sconf_valid"] = np.zeros(nb_exp)

    record_loss["loss_d_train"] = np.zeros(nb_exp)
    record_loss["loss_d_valid"] = np.zeros(nb_exp)

    record_loss["loss_raman_train"] = np.zeros(nb_exp)
    record_loss["loss_raman_valid"] = np.zeros(nb_exp)

    record_loss["loss_train"] = np.zeros(nb_exp)
    record_loss["loss_valid"] = np.zeros(nb_exp)

    for i in range(nb_exp):

        name = "./model/candidates/l"+str(nb_layers)+"_n"+str(nb_neurons)+"_p"+str(p_drop)+"_m"+str(i)+".pth"
        record_loss.loc[i,"name"] = "l"+str(nb_layers)+"_n"+str(nb_neurons)+"_p"+str(p_drop)+"_m"+str(i)+".pth"

        # declaring model
        neuralmodel = model.model(ds.x_visco_train.shape[1],nb_neurons,nb_layers,ds.nb_channels_raman,
                                  p_drop=p_drop, activation_function = torch.nn.GELU())
        neuralmodel.load_state_dict(torch.load(name, map_location='cpu'))
        neuralmodel.to('cpu')
        neuralmodel.eval()

        # PREDICTIONS
        with torch.set_grad_enabled(False):
            # train
            y_ag_pred_train = neuralmodel.ag(ds.x_visco_train,ds.T_visco_train)
            y_myega_pred_train = neuralmodel.myega(ds.x_visco_train,ds.T_visco_train)
            y_am_pred_train = neuralmodel.am(ds.x_visco_train,ds.T_visco_train)
            y_cg_pred_train = neuralmodel.cg(ds.x_visco_train,ds.T_visco_train)
            y_tvf_pred_train = neuralmodel.tvf(ds.x_visco_train,ds.T_visco_train)
            y_raman_pred_train = neuralmodel.raman_pred(ds.x_raman_train)
            y_density_pred_train = neuralmodel.density_glass(ds.x_density_train)
            y_entro_pred_train = neuralmodel.sctg(ds.x_entro_train)
            y_ri_pred_train = neuralmodel.sellmeier(ds.x_ri_train, ds.lbd_ri_train)

            # valid
            y_ag_pred_valid = neuralmodel.ag(ds.x_visco_valid,ds.T_visco_valid)
            y_myega_pred_valid = neuralmodel.myega(ds.x_visco_valid,ds.T_visco_valid)
            y_am_pred_valid = neuralmodel.am(ds.x_visco_valid,ds.T_visco_valid)
            y_cg_pred_valid = neuralmodel.cg(ds.x_visco_valid,ds.T_visco_valid)
            y_tvf_pred_valid = neuralmodel.tvf(ds.x_visco_valid,ds.T_visco_valid)
            y_raman_pred_valid = neuralmodel.raman_pred(ds.x_raman_valid)
            y_density_pred_valid = neuralmodel.density_glass(ds.x_density_valid)
            y_entro_pred_valid = neuralmodel.sctg(ds.x_entro_valid)
            y_ri_pred_valid = neuralmodel.sellmeier(ds.x_ri_valid, ds.lbd_ri_valid)

            # Compute Loss

            # train
            record_loss.loc[i,"loss_ag_train"] = np.sqrt(criterion(y_ag_pred_train, ds.y_visco_train).item())
            record_loss.loc[i,"loss_myega_train"]  = np.sqrt(criterion(y_myega_pred_train, ds.y_visco_train).item())
            record_loss.loc[i,"loss_am_train"]  = np.sqrt(criterion(y_am_pred_train, ds.y_visco_train).item())
            record_loss.loc[i,"loss_cg_train"]  = np.sqrt(criterion(y_cg_pred_train, ds.y_visco_train).item())
            record_loss.loc[i,"loss_tvf_train"]  = np.sqrt(criterion(y_tvf_pred_train, ds.y_visco_train).item())
            record_loss.loc[i,"loss_raman_train"]  = np.sqrt(criterion(y_raman_pred_train,ds.y_raman_train).item())
            record_loss.loc[i,"loss_d_train"]  = np.sqrt(criterion(y_density_pred_train,ds.y_density_train).item())
            record_loss.loc[i,"loss_Sconf_train"]  = np.sqrt(criterion(y_entro_pred_train,ds.y_entro_train).item())
            record_loss.loc[i,"loss_ri_train"]  = np.sqrt(criterion(y_ri_pred_train,ds.y_ri_train).item())

            # validation
            record_loss.loc[i,"loss_ag_valid"] = np.sqrt(criterion(y_ag_pred_valid, ds.y_visco_valid).item())
            record_loss.loc[i,"loss_myega_valid"] = np.sqrt(criterion(y_myega_pred_valid, ds.y_visco_valid).item())
            record_loss.loc[i,"loss_am_valid"] = np.sqrt(criterion(y_am_pred_valid, ds.y_visco_valid).item())
            record_loss.loc[i,"loss_cg_valid"]  = np.sqrt(criterion(y_cg_pred_valid, ds.y_visco_valid).item())
            record_loss.loc[i,"loss_tvf_valid"]  = np.sqrt(criterion(y_tvf_pred_valid, ds.y_visco_valid).item())
            record_loss.loc[i,"loss_raman_valid"] = np.sqrt(criterion(y_raman_pred_valid,ds.y_raman_valid).item())
            record_loss.loc[i,"loss_d_valid"] = np.sqrt(criterion(y_density_pred_valid,ds.y_density_valid).item())
            record_loss.loc[i,"loss_Sconf_valid"] = np.sqrt(criterion(y_entro_pred_valid,ds.y_entro_valid).item())
            record_loss.loc[i,"loss_ri_valid"]  = np.sqrt(criterion(y_ri_pred_valid,ds.y_ri_valid).item())

        record_loss.loc[i,"loss_train"] = (record_loss.loc[i,"loss_ag_train"] +
                                             record_loss.loc[i,"loss_myega_train"] +
                                             record_loss.loc[i,"loss_am_train"] +
                                             record_loss.loc[i,"loss_cg_train"] +
                                             record_loss.loc[i,"loss_tvf_train"] +
                                             raman_scale*record_loss.loc[i,"loss_raman_train"] +
                                             density_scale*record_loss.loc[i,"loss_d_train"] +
                                             entro_scale*record_loss.loc[i,"loss_Sconf_train"] +
                                             ri_scale*record_loss.loc[i,"loss_ri_train"])

        record_loss.loc[i,"loss_valid"] = (record_loss.loc[i,"loss_ag_valid"] +
                                             record_loss.loc[i,"loss_myega_valid"] +
                                             record_loss.loc[i,"loss_am_valid"] +
                                             record_loss.loc[i,"loss_cg_valid"] +
                                             record_loss.loc[i,"loss_tvf_valid"] +
                                             raman_scale*record_loss.loc[i,"loss_raman_valid"] +
                                             density_scale*record_loss.loc[i,"loss_d_valid"] +
                                             entro_scale*record_loss.loc[i,"loss_Sconf_valid"] +
                                             ri_scale*record_loss.loc[i,"loss_ri_valid"])

        record_loss.loc[i,"loss_all"] = (record_loss.loc[i,"loss_train"]+record_loss.loc[i,"loss_valid"])/2.0

    # Get the 10 best recorded
    best_recorded = record_loss.nsmallest(30,"loss_valid")

    # Copy the content of
    # source to destination
    for i in best_recorded.loc[:,"name"]:
        shutil.copyfile("./model/candidates/"+i, "./model/best/"+i)

    best_recorded.to_csv("./model/best/best_list.csv")
    
if __name__ == "__main__":
    #train_candidates()
    detect_save(nb_exp = 41)
