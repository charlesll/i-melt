import Flux
using Flux.Tracker, Statistics, DelimitedFiles
using Flux.Tracker: Params, gradient, update!
using Flux: gpu, Dense, BatchNorm, relu, Chain, ADAM, RMSProp, Descent, params, throttle, Dropout, @epochs, testmode!

using Base.Iterators: repeated

using DataFrames, DataFramesMeta, ExcelFiles, MLDataUtils, HDF5, ProgressMeter, Plots, LinearAlgebra, NNlib
pyplot() # pyplot backend

using BSON: @save, @load

include("flux_functions.jl")
include("flux_train.jl")

dataset="./data/DataSet_0p10val.hdf5"
mod_path_out= "/home/charles/architecture_exp/"

for i = 1:2
    nb_layers = rand(1:6)
    nb_neurons = rand(2:400)
    nameout = "_"*string(nb_neurons)*"_neurons_"*string(nb_layers)*"_layers"
    train_nn(dataset, mod_path_out, nameout, nb_layers=nb_layers, nb_neurons=nb_neurons,max_epoch=20, pretraining=true)
end
