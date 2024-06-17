#!/usr/bin/env python
# coding: utf-8

########## Calling relevant libraries ##########
import  h5py, os

# local imports
import imelt

import numpy as np
import pandas as pd
import rampy as rp
import matplotlib.pyplot as plt

import sklearn.model_selection as model_selection

#
# function definition
#
def prepare_raman(my_liste, output_file, include_embargo=False, rand_state=67, generate_figures=False):
    """prepare the raman dataset for the ML model"""
    # preprocess the spectra
    spectra_long = imelt.preprocess_raman(my_liste, generate_figures=generate_figures)

    # control dataset
    my_liste = imelt.chimie_control(my_liste)

    # add descriptors, then train-valid-test split
    X = imelt.descriptors(my_liste.loc[:,["sio2","al2o3","na2o","k2o","mgo","cao"]]).values
    X_train, X_vt, y_train, y_vt = model_selection.train_test_split(X,spectra_long.T,test_size=0.20, random_state=rand_state) # train-test split
    X_valid, X_test, y_valid, y_test = model_selection.train_test_split(X_vt,y_vt,test_size=0.5, random_state=rand_state) # train-test split
    
    # same thing for embargoed files
    if include_embargo == True:
        liste_eb = pd.read_excel("./data/Database.xlsx", "RAMAN_EMBARGO")
        liste_eb = imelt.chimie_control(liste_eb)
        spectra_embargo = imelt.preprocess_raman(liste_eb, generate_figures=generate_figures)
        X2 = imelt.descriptors(liste_eb.loc[:,["sio2","al2o3","na2o","k2o","mgo","cao"]]).values
        X_train2, X_test2, y_train2, y_test2 = model_selection.train_test_split(X2,spectra_embargo.T,test_size=0.15, random_state=rand_state) # train-test split
        
        # concatenate the datasets
        X_train = np.concatenate((X_train, X_train2), axis=0)
        X_test = np.concatenate((X_test, X_test2), axis=0)
        y_train = np.concatenate((y_train, y_train2), axis=0)
        y_test = np.concatenate((y_test, y_test2), axis=0)

    # save spectra in HDF5
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('X_raman_train', data=X_train)
        f.create_dataset('X_raman_valid', data=X_valid)
        f.create_dataset('X_raman_test', data=X_test)
        f.create_dataset('y_raman_train', data=y_train)
        f.create_dataset('y_raman_valid', data=y_valid)
        f.create_dataset('y_raman_test', data=y_test)

def prepare_cp(dataset,output_file, rand_state=60):
    """prepare the dataset of liquid heat capacity for the ML model"""
    # control dataset
    dataset = imelt.chimie_control(dataset)
    
    # train-valid-test group stratified split
    # 90-5-5
    train_, valid_, test_ = imelt.stratified_group_splitting(dataset, "Name", verbose = True, random_state = rand_state, n_splits=10)
    
    # grab what we are interested in
    X_cpl_train = imelt.descriptors(train_.loc[:, ["sio2","al2o3","na2o","k2o","mgo","cao"]])
    T_cpl_train = train_.loc[:, ["T"]]
    y_cpl_train = train_.loc[:, ["Cp_l"]]

    X_cpl_valid = imelt.descriptors(valid_.loc[:, ["sio2","al2o3","na2o","k2o","mgo","cao"]])
    T_cpl_valid = valid_.loc[:, ["T"]]
    y_cpl_valid = valid_.loc[:, ["Cp_l"]]

    X_cpl_test = imelt.descriptors(test_.loc[:, ["sio2","al2o3","na2o","k2o","mgo","cao"]])
    T_cpl_test = test_.loc[:, ["T"]]
    y_cpl_test = test_.loc[:, ["Cp_l"]]
    
    # Cp_l is only used as a regularization parameter during training
    # dataset is too small for ML
    # therefore no splitting is done here
    # writing the data in HDF5 file for later call
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('X_cpl_train', data=X_cpl_train)
        f.create_dataset('T_cpl_train', data=T_cpl_train)
        f.create_dataset('y_cpl_train',  data=y_cpl_train)

        # same for validation
        f.create_dataset('X_cpl_valid', data=X_cpl_valid)
        f.create_dataset('T_cpl_valid', data=T_cpl_valid)
        f.create_dataset('y_cpl_valid',  data=y_cpl_valid)

        # same for test
        f.create_dataset('X_cpl_test', data=X_cpl_test)
        f.create_dataset('T_cpl_test', data=T_cpl_test)
        f.create_dataset('y_cpl_test',  data=y_cpl_test)
        
    print("Done.")

def prepare_density(dataset, output_file, rand_state=60):
    """prepare the dataset of glass density for the ML model"""
    
    # control dataset
    dataset = imelt.chimie_control(dataset)

    # train-valid-test split
    # 80-10-10
    train_, valid_, test_ = imelt.stratified_group_splitting(dataset, "Name", verbose = True, random_state = rand_state)
    
    # grab good X columns and add descriptors
    X_train = imelt.descriptors(train_.loc[:,["sio2","al2o3","na2o","k2o","mgo","cao"]]).values
    X_valid = imelt.descriptors(valid_.loc[:,["sio2","al2o3","na2o","k2o","mgo","cao"]]).values
    X_test = imelt.descriptors(test_.loc[:,["sio2","al2o3","na2o","k2o","mgo","cao"]]).values

    # grab the good y values
    y_train = train_.loc[:, ["d"]].values.reshape(-1,1)
    y_valid = valid_.loc[:, ["d"]].values.reshape(-1,1)
    y_test = test_.loc[:, ["d"]].values.reshape(-1,1)
     
    # writing the data in HDF5 file for later call
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('X_density_train', data=X_train)
        f.create_dataset('X_density_valid', data=X_valid)
        f.create_dataset('X_density_test',  data=X_test)

        f.create_dataset('y_density_train', data=y_train)
        f.create_dataset('y_density_valid', data=y_valid)
        f.create_dataset('y_density_test',  data=y_test)
        
    print("Done.")
 
def prepare_liquidus(dataset, output_file, rand_state=60):
    """prepare the dataset of glass density for the ML model"""
    
    # control dataset
    dataset = imelt.chimie_control(dataset)

    # train-valid-test split
    # 80-10-10
    train_, valid_, test_ = imelt.stratified_group_splitting(dataset, "Name", verbose = True, random_state = rand_state)
    
    # grab good X columns and add descriptors
    X_train = imelt.descriptors(train_.loc[:,["sio2","al2o3","na2o","k2o","mgo","cao"]]).values
    X_valid = imelt.descriptors(valid_.loc[:,["sio2","al2o3","na2o","k2o","mgo","cao"]]).values
    X_test = imelt.descriptors(test_.loc[:,["sio2","al2o3","na2o","k2o","mgo","cao"]]).values

    # grab the good y values
    y_train = train_.loc[:, ["T_K"]].values.reshape(-1,1)
    y_valid = valid_.loc[:, ["T_K"]].values.reshape(-1,1)
    y_test = test_.loc[:, ["T_K"]].values.reshape(-1,1)
     
    # writing the data in HDF5 file for later call
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('X_liquidus_train', data=X_train)
        f.create_dataset('X_liquidus_valid', data=X_valid)
        f.create_dataset('X_liquidus_test',  data=X_test)

        f.create_dataset('y_liquidus_train', data=y_train)
        f.create_dataset('y_liquidus_valid', data=y_valid)
        f.create_dataset('y_liquidus_test',  data=y_test)
        
    print("Done.")

def prepare_abbe(dataset, output_file, rand_state=60):
    """prepare the dataset of glass Abbe Number for the ML model"""
    
    # control dataset
    dataset = imelt.chimie_control(dataset)

    # train-valid-test split
    # 80-10-10
    train_, valid_, test_ = imelt.stratified_group_splitting(dataset, "Name", verbose = True, random_state = rand_state)
    
    # grab good X columns and add descriptors
    X_train = imelt.descriptors(train_.loc[:,["sio2","al2o3","na2o","k2o","mgo","cao"]]).values
    X_valid = imelt.descriptors(valid_.loc[:,["sio2","al2o3","na2o","k2o","mgo","cao"]]).values
    X_test = imelt.descriptors(test_.loc[:,["sio2","al2o3","na2o","k2o","mgo","cao"]]).values

    # grab the good y values
    y_train = train_.loc[:, ["AbbeNumber"]].values.reshape(-1,1)
    y_valid = valid_.loc[:, ["AbbeNumber"]].values.reshape(-1,1)
    y_test = test_.loc[:, ["AbbeNumber"]].values.reshape(-1,1)
     
    # writing the data in HDF5 file for later call
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('X_abbe_train', data=X_train)
        f.create_dataset('X_abbe_valid', data=X_valid)
        f.create_dataset('X_abbe_test',  data=X_test)

        f.create_dataset('y_abbe_train', data=y_train)
        f.create_dataset('y_abbe_valid', data=y_valid)
        f.create_dataset('y_abbe_test',  data=y_test)
        
    print("Done.")

def prepare_elastic(dataset, output_file, rand_state=60):
    """prepare the dataset of glass elastic modulus for the ML model"""
    
    # control dataset
    dataset = imelt.chimie_control(dataset)

    # train-valid-test split
    # 80-10-10
    train_, valid_, test_ = imelt.stratified_group_splitting(dataset, "Name", verbose = True, random_state = rand_state)
    
    # grab good X columns and add descriptors
    X_train = imelt.descriptors(train_.loc[:,["sio2","al2o3","na2o","k2o","mgo","cao"]]).values
    X_valid = imelt.descriptors(valid_.loc[:,["sio2","al2o3","na2o","k2o","mgo","cao"]]).values
    X_test = imelt.descriptors(test_.loc[:,["sio2","al2o3","na2o","k2o","mgo","cao"]]).values

    # grab the good y values
    y_train = train_.loc[:, ["EM"]].values.reshape(-1,1)
    y_valid = valid_.loc[:, ["EM"]].values.reshape(-1,1)
    y_test = test_.loc[:, ["EM"]].values.reshape(-1,1)
     
    # writing the data in HDF5 file for later call
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('X_elastic_train', data=X_train)
        f.create_dataset('X_elastic_valid', data=X_valid)
        f.create_dataset('X_elastic_test',  data=X_test)

        f.create_dataset('y_elastic_train', data=y_train)
        f.create_dataset('y_elastic_valid', data=y_valid)
        f.create_dataset('y_elastic_test',  data=y_test)
        
    print("Done.")

def prepare_cte(dataset, output_file, rand_state=60):
    """prepare the dataset of glass CTE for the ML model
    
    CTE are scaled with a 1e6 coefficient to avoid numerical issues"""
    
    # control dataset
    dataset = imelt.chimie_control(dataset)

    # train-valid-test split
    # 80-10-10
    train_, valid_, test_ = imelt.stratified_group_splitting(dataset, "Name", verbose = True, random_state = rand_state)
    
    # grab good X columns and add descriptors
    X_train = imelt.descriptors(train_.loc[:,["sio2","al2o3","na2o","k2o","mgo","cao"]]).values
    X_valid = imelt.descriptors(valid_.loc[:,["sio2","al2o3","na2o","k2o","mgo","cao"]]).values
    X_test = imelt.descriptors(test_.loc[:,["sio2","al2o3","na2o","k2o","mgo","cao"]]).values

    # grab the good y values
    y_train = train_.loc[:, ["CTE_scaled"]].values.reshape(-1,1)
    y_valid = valid_.loc[:, ["CTE_scaled"]].values.reshape(-1,1)
    y_test = test_.loc[:, ["CTE_scaled"]].values.reshape(-1,1)
     
    # writing the data in HDF5 file for later call
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('X_cte_train', data=X_train)
        f.create_dataset('X_cte_valid', data=X_valid)
        f.create_dataset('X_cte_test',  data=X_test)

        f.create_dataset('y_cte_train', data=y_train)
        f.create_dataset('y_cte_valid', data=y_valid)
        f.create_dataset('y_cte_test',  data=y_test)
        
    print("Done.")

def prepare_viscosity(dataset,output_file, rand_state=67, include_embargo=False):
    """prepare the dataset of glass-forming melt viscosity for the ML model"""
    print('Reading data...')
    # reading the Pandas dataframe
    dataset = imelt.chimie_control(dataset)

    ####
    # viscosity
    # train-valid-test group stratified split
    # 80-10-10
    ####
    train_, valid_, test_ = imelt.stratified_group_splitting(dataset, "Name", verbose = True, random_state = rand_state)
    
    if include_embargo == True:
        dataset_eb = imelt.chimie_control(pd.read_excel("./data/Database.xlsx", sheet_name="VISCO_EMBARGO"))
        train_ = pd.concat([train_, dataset_eb], axis=0)

    # grab good X columns and add descriptors
    X_train = imelt.descriptors(train_.loc[:,["sio2","al2o3","na2o","k2o","mgo","cao"]]).values
    X_valid = imelt.descriptors(valid_.loc[:,["sio2","al2o3","na2o","k2o","mgo","cao"]]).values
    X_test = imelt.descriptors(test_.loc[:,["sio2","al2o3","na2o","k2o","mgo","cao"]]).values

    # temperature values
    T_train = train_.loc[:,"T"].values.reshape(-1,1)
    T_valid = valid_.loc[:,"T"].values.reshape(-1,1)
    T_test = test_.loc[:,"T"].values.reshape(-1,1)

    # grab the good y values
    y_train = train_.loc[:, ["viscosity"]].values.reshape(-1,1)
    y_valid = valid_.loc[:, ["viscosity"]].values.reshape(-1,1)
    y_test = test_.loc[:, ["viscosity"]].values.reshape(-1,1)
   
    ####
    # Entropy
    # train-valid-test split
    # 80-10-10
    ####

    # we drop all rows without entropy values and get only one value per composition
    dataset_entropy = dataset.dropna(subset=['Sc']).copy()
    dataset_entropy.drop_duplicates(subset ="Name",keep = "first", inplace = True)
    dataset_entropy["class"] = imelt.class_data(dataset_entropy)

    # 80-10-10 split
    train_entropy, tv_entropy = model_selection.train_test_split(dataset_entropy,
                                                                 test_size=0.20, random_state=rand_state)
    test_entropy, valid_entropy = model_selection.train_test_split(tv_entropy, 
                                                                   test_size=0.5, random_state=rand_state)
    
    # we include embargoed data
    if include_embargo == True:
        dataset_entropy_eb = dataset_eb.dropna(subset=['Sc']).copy()
        dataset_entropy_eb.drop_duplicates(subset ="Name",keep = "first", inplace = True)
        train_entropy = pd.concat([train_entropy, dataset_entropy_eb], axis=0)

    X_columns = ["sio2","al2o3","na2o","k2o","mgo","cao"] # for output
    # get good columns and add descriptors
    X_entropy_train = imelt.descriptors(train_entropy.loc[:,X_columns]).values
    X_entropy_valid = imelt.descriptors(valid_entropy.loc[:,X_columns]).values
    X_entropy_test = imelt.descriptors(test_entropy.loc[:,X_columns]).values
    
    y_entropy_train = train_entropy.loc[:,"Sc"].values
    y_entropy_valid = valid_entropy.loc[:,"Sc"].values
    y_entropy_test = test_entropy.loc[:,"Sc"].values
    
    ####
    # for Tg
    # we grab the Tgs associated with the train-valid-test split of viscosity data
    # (as Tg is not used for training per se)
    ####
     
    # we drop the values at 0 (Tg not determined)
    # we drop all rows without Tg values and get only one value per composition
    train_tg = train_.loc[train_.tg != 0,["Name","sio2","al2o3","na2o","k2o","mgo","cao","tg"]]
    valid_tg = valid_.loc[valid_.tg != 0,["Name","sio2","al2o3","na2o","k2o","mgo","cao","tg"]]
    test_tg  =  test_.loc[test_.tg  != 0,["Name","sio2","al2o3","na2o","k2o","mgo","cao","tg"]]

    # drop duplicates (in place)
    train_tg.drop_duplicates(subset ="Name",keep = "first", inplace = True)
    valid_tg.drop_duplicates(subset ="Name",keep = "first", inplace = True)
    test_tg.drop_duplicates(subset ="Name",keep = "first", inplace = True)

    # add descriptors and convert to numpy and continue
    X_tg_train = imelt.descriptors(train_tg.loc[:,["sio2","al2o3","na2o","k2o","mgo","cao"]]).values
    X_tg_valid = imelt.descriptors(valid_tg.loc[:,["sio2","al2o3","na2o","k2o","mgo","cao"]]).values
    X_tg_test = imelt.descriptors(test_tg.loc[:,["sio2","al2o3","na2o","k2o","mgo","cao"]]).values

    y_tg_train = train_tg.loc[:,"tg"].values
    y_tg_valid = valid_tg.loc[:,"tg"].values
    y_tg_test = test_tg.loc[:,"tg"].values

    # Figure of the datasets
    plt.figure()
    plt.subplot(121)
    plt.plot(10000/T_train,y_train,"k.", label="train")

    plt.subplot(121)
    plt.plot(10000/T_valid,y_valid,"b.", label="valid")

    plt.subplot(121)
    plt.plot(10000/T_test,y_test,"r.", label="test")
    plt.savefig(output_file+".pdf")
    plt.close()

    print("Size of viscous training subsets:\n")
    print(X_train.shape)
    print(X_valid.shape)
    print(X_test.shape)
    
    # writing the data in HDF5 file for later call
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('X_columns', data=X_columns)#data=np.array(X_columns, dtype="S10"))

        f.create_dataset('X_entropy_train', data=X_entropy_train)
        f.create_dataset('y_entropy_train', data=y_entropy_train.reshape(len(y_entropy_train),1))
        
        f.create_dataset('X_entropy_valid', data=X_entropy_valid)
        f.create_dataset('y_entropy_valid', data=y_entropy_valid.reshape(len(y_entropy_valid),1))
        
        f.create_dataset('X_entropy_test', data=X_entropy_test)
        f.create_dataset('y_entropy_test', data=y_entropy_test.reshape(len(y_entropy_test),1))
        
        f.create_dataset('X_tg_train',data=X_tg_train)
        f.create_dataset('X_tg_valid',data=X_tg_valid)
        f.create_dataset('X_tg_test',data=X_tg_test)
        
        f.create_dataset('y_tg_train',data=y_tg_train.reshape(len(y_tg_train),1))
        f.create_dataset('y_tg_valid',data=y_tg_valid.reshape(len(y_tg_valid),1))
        f.create_dataset('y_tg_test',data=y_tg_test.reshape(len(y_tg_test),1))

        f.create_dataset('X_train', data=X_train)
        f.create_dataset('T_train', data=T_train)
        f.create_dataset('y_train', data=y_train)
        
        f.create_dataset('X_valid', data=X_valid)
        f.create_dataset('T_valid', data=T_valid)
        f.create_dataset('y_valid', data=y_valid)
        
        f.create_dataset('X_test', data=X_test)
        f.create_dataset('T_test', data=T_test)
        f.create_dataset('y_test', data=y_test)

def prepare_ri(dataset,output_file, rand_state=81):
    """prepare the optical refractive index data for the ML model"""
    
    # control dataset
    dataset = imelt.chimie_control(dataset)

    # train-valid-test split
    # 80-10-10
    train_, valid_, test_ = imelt.stratified_group_splitting(dataset, "Name", verbose = True, random_state = rand_state)
    
    # grab good X columns and add descriptors
    X_train = imelt.descriptors(train_.loc[:,["sio2","al2o3","na2o","k2o","mgo","cao"]]).values
    X_valid = imelt.descriptors(valid_.loc[:,["sio2","al2o3","na2o","k2o","mgo","cao"]]).values
    X_test = imelt.descriptors(test_.loc[:,["sio2","al2o3","na2o","k2o","mgo","cao"]]).values

    # lambda values
    lbd_train = train_.loc[:,"lbd"].values.reshape(-1,1)*1e-3
    lbd_valid = valid_.loc[:,"lbd"].values.reshape(-1,1)*1e-3
    lbd_test = test_.loc[:,"lbd"].values.reshape(-1,1)*1e-3

    # grab the good y values
    y_train = train_.loc[:, ["ri"]].values.reshape(-1,1)
    y_valid = valid_.loc[:, ["ri"]].values.reshape(-1,1)
    y_test = test_.loc[:, ["ri"]].values.reshape(-1,1)
           
    # writing the data in HDF5 file for later call
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('X_ri_train', data=X_train)
        f.create_dataset('X_ri_valid', data=X_valid)
        f.create_dataset('X_ri_test',  data=X_test)

        f.create_dataset('lbd_ri_train', data=lbd_train)
        f.create_dataset('lbd_ri_valid', data=lbd_valid)
        f.create_dataset('lbd_ri_test',  data=lbd_test)

        f.create_dataset('y_ri_train', data=y_train)
        f.create_dataset('y_ri_valid', data=y_valid)
        f.create_dataset('y_ri_test',  data=y_test)

    print("Done.")

if __name__=="__main__":

    # ask which dataset needs to be prepared:
    print("Which dataset do you want to prepare?")
    print("Type:\n    v for viscosity")
    print("    r for Raman spectroscopy")
    print("    d for density, ")
    print("    o for optical refractive index")
    print("    c for liquid heat capacity")
    print("    l for liquidus temperature")
    print("    a for Abbe Number")
    print("    e for Elastic Modulus")
    print("    cte for CTE")

    good = False
    while good == False:
        user_input = input("Enter the desired value:")
        if user_input in ["v","r","o","d","c","l","a","e","cte"]:
            good = True
    
    if user_input == "v":
        # Viscosity preparation
        print("Preparing the viscosity datasets...")
        dataset = pd.read_excel("./data/Database.xlsx",sheet_name="VISCO")
        prepare_viscosity(dataset,"./data/NKCMAS_viscosity.hdf5", rand_state=127, include_embargo=False)

    if user_input == "d":
        # Density preparation
        print("Preparing the density dataset...")
        prepare_density(pd.read_excel("./data/Database.xlsx",sheet_name="DENSITY"),"./data/NKCMAS_density.hdf5")

    if user_input == "o":
        # Refractive Index preparation
        print("Preparing the optical refractive index dataset...")
        prepare_ri(pd.read_excel("./data/Database.xlsx",sheet_name="OPTICAL"),"./data/NKCMAS_optical.hdf5", rand_state=60)

    if user_input == "r":
        # Raman spectra preparation
        print("Preparing the Raman spectra dataset...")
        prepare_raman(pd.read_excel("./data/Database.xlsx", "RAMAN"), 
                      './data/NKCMAS_Raman.hdf5', include_embargo=False,
                      generate_figures=True, rand_state=127)

    if user_input == "c":
        # Heat capacity preparation
        print("Preparing the heat capacity dataset...")
        prepare_cp(pd.read_excel("./data/Database.xlsx", "CP"), './data/NKCMAS_cp.hdf5')

    if user_input == "l":
        # liquidus preparation
        print("Preparing the liquidus dataset...")
        prepare_liquidus(pd.read_excel("./data/Database.xlsx", "Liquidus"), './data/NKCMAS_tl.hdf5')

    if user_input == "a":
        # Abbe number preparation
        print("Preparing the Abbe number dataset...")
        prepare_abbe(pd.read_excel("./data/Database.xlsx", "AbbeNumber"), './data/NKCMAS_abbe.hdf5')
    
    if user_input == "e":
        # Elastic modulus preparation
        print("Preparing the elastic modulus dataset...")
        prepare_elastic(pd.read_excel("./data/Database.xlsx", "ElasticModulus"), './data/NKCMAS_em.hdf5')
    
    if user_input == "cte":
        # CTE preparation
        print("Preparing the CTE dataset...")
        prepare_cte(pd.read_excel("./data/Database.xlsx", "CTE"), './data/NKCMAS_cte.hdf5')
