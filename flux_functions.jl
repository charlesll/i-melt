#
# DATA MANIPULATIONS
#

"""
    load_data(path_dataset::String, path_raman::String, path_density::String)

load the viscosity and Raman datasets at path_dataset and path_raman, respectively
"""
function load_data(path_data::String, path_raman::String, path_density::String;verbose=true)

    datas = Dict()
    datas["X_columns"] = h5read(path_data, "X_columns")

    datas["X_entropy_train"] = h5read(path_data, "X_entropy_train")
    datas["X_entropy_train_sc"] = h5read(path_data, "X_entropy_train_sc")

    datas["X_entropy_valid"] = h5read(path_data, "X_entropy_valid")
    datas["X_entropy_valid_sc"] = h5read(path_data, "X_entropy_valid_sc")

    datas["X_entropy_test"] = h5read(path_data, "X_entropy_test")
    datas["X_entropy_test_sc"] = h5read(path_data, "X_entropy_test_sc")

    datas["X_entropy_tv"] = [datas["X_entropy_train"] datas["X_entropy_valid"]]
    datas["X_entropy_sc_tv"] = [datas["X_entropy_train_sc"] datas["X_entropy_valid_sc"]]

    datas["X_tv"] = h5read(path_data, "X_tv")
    datas["X_tv_sc"] = h5read(path_data, "X_tv_sc")
    datas["y_tv"] = h5read(path_data, "y_tv")

    datas["X_train"] = h5read(path_data, "X_train")
    datas["X_train_sc"] = h5read(path_data, "X_train_sc")
    datas["y_train"] = h5read(path_data, "y_train")

    datas["X_valid"] = h5read(path_data, "X_valid")
    datas["X_valid_sc"] = h5read(path_data, "X_valid_sc")
    datas["y_valid"] = h5read(path_data, "y_valid")

    datas["X_test"] = h5read(path_data, "X_test")
    datas["X_test_sc"] = h5read(path_data, "X_test_sc")
    datas["y_test"] = h5read(path_data, "y_test")

    datas["X_scaler_mean"] = h5read(path_data, "X_scaler_mean")
    datas["X_scaler_var"] = h5read(path_data, "X_scaler_var")

    datas["X_raman_train"] = Float32.(h5read(path_raman,"X_raman_train"))
    datas["y_raman_train"] = Float32.((h5read(path_raman,"y_raman_train")))
    datas["X_raman_valid"] = Float32.(h5read(path_raman,"X_raman_test"))
    datas["y_raman_valid"] = Float32.((h5read(path_raman,"y_raman_test")))

    # Loading density dataset
    datas["X_density_train"] = Float32.(h5read("./data/NKAS_density.hdf5","X_density_train"))
    datas["X_density_valid"] = Float32.(h5read("./data/NKAS_density.hdf5","X_density_valid"))
    datas["X_density_test"] = Float32.(h5read("./data/NKAS_density.hdf5","X_density_test"))

    datas["y_density_train"] = Float32.(h5read("./data/NKAS_density.hdf5","y_density_train"))
    datas["y_density_valid"] = Float32.(h5read("./data/NKAS_density.hdf5","y_density_valid"))
    datas["y_density_test"] = Float32.(h5read("./data/NKAS_density.hdf5","y_density_test"))

    if verbose == true
        println("loaded")
        println("\nFeatures in X_ arrays are")
        println(datas["X_columns"])
        println("\nShape of X train and valid is")
        println(size(datas["X_tv"]))

        println("Size of Raman datasets")
        println(size(datas["X_raman_train"]))
        println(size(datas["y_raman_train"]))
        println(size(datas["X_raman_valid"]))
        println(size(datas["y_raman_valid"]))
    end

    return datas
end

"""
    prepare_datas(X_,y_)

Prepare datas
"""
function prepare_data(X_,y_)

    y = reshape(y_[:],1,length(y_))

    x = X_[1:4,:]
    ap = reshape(X_[8,:],1,size(X_,2))
    b = reshape(X_[9,:],1,size(X_,2))
    T = reshape(X_[10,:],1,size(X_,2))
    sc = reshape(X_[11,:],1,size(X_,2))
    tg = reshape(X_[12,:],1,size(X_,2))
    return Float32.(x), Float32.(y), Float32.(T), Float32.(ap), Float32.(b), Float32.(sc), Float32.(tg)
end

"""
    gkfolds(X_, y_, idx_label; k = 5)

K-fold data preparation
"""
function gkfolds(X_, y_, idx_label; k = 5)

    dd = kfolds(shuffleobs(unique(X_[idx_label,:])), k = k);

    out = []

    for j = 1:k
        train_lab, vald_lab = dd[j]
        train_idx = Int64[]
        valid_idx = Int64[]

        for i = 1:size(X_,2)
            if findall(X_[idx_label,i] .== train_lab) != []
                push!(train_idx, i)
            else
                push!(valid_idx, i)
            end
        end

        push!(out,((X_[:,train_idx],y_[train_idx]),(X_[:,valid_idx],y_[valid_idx])))

    end

    return out
end

#
# Cp calculations
#
at_gfu(x) = 3.0.*x[1,:] .+ 5.0.*x[2,:] + 3.0.*x[3,:] + 3.0.*x[4,:]# + 2*MgO + 2*MgO
aCpl(x) = 81.37.*x[1,:] .+ 27.21.*x[2,:] .+ 100.6.*x[3,:]  .+ 50.13.*x[4,:] .+ x[1,:].*(x[4,:].*x[4,:]).*151.7

ap(x) = reshape(aCpl(x) - 3.0.*8.314.*at_gfu(x),1,size(x,2))
b(x) = reshape(0.0943.*x[2,:] + 0.01578.*x[4,:],1,size(x,2)) #bCpl

#ST
# Function for initial bias values
#

function thousands(size)
    return ones(size).*log.(1000.0)
    end

function tens(size)
    return ones(size).*log.(10.0)
end

function init_both(dims)
    return ones(dims).*[log.(1000.);log.(10.);log.(10.); log.(2.3)]
end

init_random(dims...) = randn(Float32, dims...) .* [5.;10.]

#
# Function for extracting parameters from the network
#
function tg(x,network)
    return reshape(exp.(network(x[1:4,:])[1,:]),1,size(x,2))
end

function ScTg(x,Ae,ap,network)
    return ap./(fragility(x,network)./(12.0.-Ae).-1)
    #return reshape(exp.(network(x[1:4,:])[2,:]),1,size(x,2))
end

function fragility(x,network)
    return reshape(exp.(network(x[1:4,:])[3,:]),1,size(x,2))
end

function density(x,network)
    return reshape(exp.(network(x[1:4,:])[4,:]),1,size(x,2))
end

#
# Thermodynamic equations : Adam and Gibbs model
#

function Be(x,Ae,ap,network)
    return (12.0.-Ae).*(tg(x,network) .* ScTg(x,Ae,ap,network))
end

function dCp(x, T, ap, b, network)
    return ap.*(log.(T).-log.(tg(x,network))) .+ b.*(T.-tg(x,network))
end

# AG EQUATION
function ag(x, T, ap, b, network, Ae)
    return Ae .+ Be(x,Ae,ap,network) ./ (T.* (ScTg(x,Ae,ap,network) .+ dCp(x, T, ap, b,network)))
end

# MYEGA EQUATION
function myega(x, T, network, Ae)
    return Ae .+ (12.0 .- Ae).*(tg(x,network)./T).*exp.((fragility(x,network)./(12.0.-Ae).-1.0).*(tg(x,network)./T.-1.0))
end

#
# LOSS FUNCTIONS
#

function mse(yp, y)
    return sqrt(sum((yp .- y).^2)./size(y, 2))
end

function loss_n(x, T, ap, b, y_target,network, Ae)
    return mse(ag(x, T, ap, b,network, Ae), y_target) # viscosity AG
end

function loss_n_myega(x, T, y_target,network, Ae)
    return mse(myega(x, T,network, Ae), y_target) # viscosity MYEGA
end

function loss_n_tvf(x, T, y_target,network)
    return mse(tvf(x,T,network),y_target) # viscosity TVF
end

function loss_sc(x,sc,Ae,ap,network)
    return mse(ScTg(x,Ae,ap,network),sc) # Configurational entropy
end

function loss_tg(x,tg_target,network)
    return mse(tg(x,network),tg_target) # glass transition T
end

function loss_raman(x,raman_target,network)
    return mse(network(x),raman_target) # Raman spectra
end

function loss_density(x,density_target,network)
    return mse(density(x,network),density_target)
end

function loss_tg_sc_d(x,tg_target,sc_target,x_d, d_target,nns; L2_norm = 0.001, s_scale = 100.0, tg_scale = 1.0, d_scale = 1000.)
        return tg_scale.*loss_tg(x,tg_target,nns) .+ s_scale.*loss_sc(x,sc_target,nns) .+ d_scale.*loss_density(x_d,d_target,nns) .+ L2_norm*sum(norm, params(nns))# Add this to your loss
end
