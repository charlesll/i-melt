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

mod_path_out= "./model/"

liste_dataset = ["./data/DataSet_0p10val.hdf5",
                 "./data/DataSet_0p20val.hdf5",
                 "./data/DataSet_0p30val.hdf5",
                 "./data/DataSet_0p50val.hdf5",
                 "./data/DataSet_0p70val.hdf5"]

liste_suffix = ["_1","_2","_3","_4","_5"]

for i = 1:length(liste_dataset)
    for j = 1:length(liste_suffix)
        train_nn(liste_dataset[i], mod_path_out, liste_suffix[j])
    end
end
