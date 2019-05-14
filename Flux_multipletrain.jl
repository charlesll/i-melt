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

liste_suffix = ["4x100_1","4x100_2","4x100_3","4x100_4","4x100_5","4x100_6"]

for i in 1:length(liste_suffix)
    println("###########################\n ######   RUN "*liste_suffix[i]*"\n###########################")
    train_nn(dataset, mod_path_out, liste_suffix[i],
    nb_neurons = 100, p_drop = 0.3, nb_layers = 4,
    max_epoch = 5000,pretraining=true,figures=true)
end
