import Flux
using Flux.Tracker, Statistics, DelimitedFiles
using Flux.Tracker: Params, gradient, update!
using Flux: gpu, Dense, BatchNorm, relu, Chain, ADAM, RMSProp, Descent, params, throttle, Dropout, @epochs, testmode!

using Base.Iterators: repeated

using MLDataUtils, HDF5, ProgressMeter, Plots, LinearAlgebra, NNlib
pyplot() # pyplot backend

using BSON: @save, @load

include("flux_functions.jl")
include("flux_train.jl")

mod_path_out= "./model/"

dataset="./data/DataSet_0p20val.hdf5"

train_nn(dataset, mod_path_out, "_100x3",
nb_neurons = 100, p_drop = 0.3, nb_layers = 3,
max_epoch=6000,pretraining=true,figures=true)
