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

dataset="./data/DataSet_0p10val.hdf5"

train_nn(dataset, mod_path_out, "_test",max_epoch=1000,pretraining=true,figures=true)
