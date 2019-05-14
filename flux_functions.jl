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
    datas["y_entropy_train"] = h5read(path_data, "y_entropy_train")

    datas["X_entropy_valid"] = h5read(path_data, "X_entropy_valid")
    datas["y_entropy_valid"] = h5read(path_data, "y_entropy_valid")

    datas["X_entropy_test"] = h5read(path_data, "X_entropy_test")
    datas["y_entropy_test"] = h5read(path_data, "y_entropy_test")

    #datas["X_tv"] = h5read(path_data, "X_tv")
    #datas["X_tv_sc"] = h5read(path_data, "X_tv_sc")
    #datas["y_tv"] = h5read(path_data, "y_tv")

    datas["X_train"] = h5read(path_data, "X_train")
    #datas["X_train_sc"] = h5read(path_data, "X_train_sc")
    datas["y_train"] = h5read(path_data, "y_train")

    datas["X_valid"] = h5read(path_data, "X_valid")
    #datas["X_valid_sc"] = h5read(path_data, "X_valid_sc")
    datas["y_valid"] = h5read(path_data, "y_valid")

    datas["X_test"] = h5read(path_data, "X_test")
    #datas["X_test_sc"] = h5read(path_data, "X_test_sc")
    datas["y_test"] = h5read(path_data, "y_test")

    #datas["X_scaler_mean"] = h5read(path_data, "X_scaler_mean")
    #datas["X_scaler_var"] = h5read(path_data, "X_scaler_var")

    # Loading viscous Tg
    datas["X_tg_train"] = h5read(path_data,"X_tg_train")
    datas["X_tg_valid"] = h5read(path_data,"X_tg_valid")
    datas["X_tg_test"]  = h5read(path_data,"X_tg_test")

    datas["y_tg_train"] = h5read(path_data,"y_tg_train")
    datas["y_tg_valid"] = h5read(path_data,"y_tg_valid")
    datas["y_tg_test"]  = h5read(path_data,"y_tg_test")

    # Lading Raman dataset
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

Prepare datas for the viscosity dataset
"""
function prepare_data(X_, y_)

    y = reshape(y_[:],1,length(y_)) # we make sure this is at the good shape

    x = X_[1:4,:]
    T = reshape(X_[5,:],1,size(X_,2))
    ap = ap(x)
    b = b(x)
    return Float32.(x), Float32.(y), Float32.(T), Float32.(ap), Float32.(b)
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
"""
    at_gfu(x)

calculate atom per gram formula unit

assumes rows are sio2 al2o3 na2o k2o
"""
at_gfu(x) = 3.0.*x[1,:] .+ 5.0.*x[2,:] + 3.0.*x[3,:] + 3.0.*x[4,:]

"""
    aCpl(x)

calculate term a in equation Cpl = qCpl + bCpl*T
"""
aCpl(x) = 81.37.*x[1,:] .+ 27.21.*x[2,:] .+ 100.6.*x[3,:]  .+ 50.13.*x[4,:] .+ x[1,:].*(x[4,:].*x[4,:]).*151.7

"""
    b(x)

calculate term b in equation Cpl = aCpl + b*T
"""
b(x) = reshape(0.09428.*x[2,:] + 0.01578.*x[4,:],1,size(x,2)) #bCpl

"""
    ap(x)

calculate term ap in equation dS = ap ln(T/Tg) + b(T-Tg)
"""
ap(x) = reshape(aCpl(x) - 3.0.*8.314.*at_gfu(x),1,size(x,2))

#
# Function for initial bias values
#

"""
    init_both(dims)

bias initialisation to values close to physical ones
"""
init_both(dims) = ones(dims).*[log.(1000.);log.(10.0);log.(30.0); log.(2.3)]

#
# Function for extracting parameters from the network
#
"""
    tg(x,network)

predict glass transition temperature given entries X and neural network
"""
tg(x,network) = reshape(exp.(network(x[1:4,:])[1,:]),1,size(x,2))

"""
    ScTg(x,network)

predict configurational entropy given entries X and neural network
"""
ScTg(x, network) = reshape(exp.(network(x[1:4,:])[2,:]),1,size(x,2))

"""
    fragility(x,network)

predict fragility given entries X and neural network
"""
fragility(x,network) = reshape(exp.(network(x[1:4,:])[3,:]),1,size(x,2))

"""
    density(x,network)

predict density given entries X and neural network
"""
 density(x,network) = reshape(exp.(network(x[1:4,:])[4,:]),1,size(x,2))

#
# Thermodynamic equations : Adam and Gibbs model
#
"""
    Be(x,network, Ae)

predict the Be term of the Adam-Gibbs equation given entries X, neural network and Ae
"""
Be(x,network, Ae) = (12.0.-Ae).*(tg(x,network) .* ScTg(x,network))

"""
    dCp(x, T, ap, b, network)

predict the delta_CpConf term of the Adam-Gibbs equation given entries X, temperature T, Cp terms ap and b, and neural network
"""
dCp(x, T, ap, b, network) = ap.*(log.(T).-log.(tg(x,network))) .+ b.*(T.-tg(x,network))

"""
    ag(x, T, ap, b, network, Ae)

predict viscosity using the Adam-Gibbs equation, given entries X, temperature T, Cp terms ap and b, neural network and Ae
"""
ag(x, T, ap, b, network, Ae) = Ae .+ Be(x,network, Ae) ./ (T.* (ScTg(x,network) .+ dCp(x, T, ap, b,network)))

"""
    myega(x, T, network, Ae)

predict viscosity using the MYEGA equation, given entries X, temperature T, neural network and Ae
"""
myega(x, T, network, Ae) = Ae .+ (12.0 .- Ae).*(tg(x,network)./T).*exp.((fragility(x, network)./(12.0.-Ae).-1.0).*(tg(x,network)./T.-1.0))

"""
    am(x, T, network, Ae)

predict viscosity using the Avramov-Mitchell equation, given entries X, temperature T, neural network and Ae
"""
function am(x, T, network, Ae)
    return Ae .+ (12.0 .- Ae).*(tg(x,network)./T).^(fragility(x, network)./12.0)
end

#
# LOSS FUNCTIONS
#
"""
    mse(yp, y)

root mean square error between yp and y

careful with the dimension... should be the same!
"""
mse(yp, y) = return sqrt(sum((yp .- y).^2)./size(y, 2))

"""
    loss_n_ag(x, T, ap, b, y_target,network, Ae)

Adam-Gibbs viscosity loss function
"""
loss_n_ag(x, T, ap, b, y_target,network, Ae) = mse(ag(x, T, ap, b,network, Ae), y_target)

"""
    loss_n_myega(x, T, y_target, network, Ae)

MYEGA viscosity loss function
"""
loss_n_myega(x, T, y_target, network, Ae) = mse(myega(x, T, network, Ae), y_target)

"""
    loss_n_am(x, T, y_target, network, Ae)

AM viscosity loss function
"""
loss_n_am(x, T, y_target, network, Ae) = mse(am(x, T, network, Ae), y_target)

"""
    loss_n_tvf(x, T, y_target,network)

VFT viscosity loss function
"""
loss_n_tvf(x, T, y_target,network) = mse(tvf(x,T,network),y_target) # viscosity TVF

"""
    loss_sc(X_, Sc_, nns)

Sconf(Tg) loss function
"""
loss_sc(X_, Sc_, nns) = mse(ScTg(X_, nns),Sc_)

"""
    loss_tg(x,tg_target,network)

Glass transition temperature Tg loss function
"""
loss_tg(x,tg_target,network) = mse(tg(x,network),tg_target) # glass transition T

"""
    loss_raman(x,raman_target,network)

Raman spectra loss function
"""
loss_raman(x,raman_target,network) = mse(network(x),raman_target) # Raman spectra

"""
    loss_density(x,density_target,network)

Density loss function
"""
loss_density(x,density_target,network) = mse(density(x,network),density_target)

"""
    loss_tg_d_sc(x_tg, y_tg, x_d, y_d, x_s, y_s, nns; L2_norm = 0.01, K_s = 100.0, K_t = 1.0, K_d = 1000.)

For pretraining on Tg, Sconf(Tg) and density
"""
function loss_tg_d_sc(x_tg, y_tg, x_d, y_d, x_s, y_s, nns; L2_norm = 0.01, K_s = 1000.0, K_t = 1.0, K_d = 10000.)
        return K_t.*loss_tg(x_tg,y_tg,nns)
        .+ K_d.*loss_density(x_d,y_d,nns)
        .+ K_s.*loss_sc(x_s, y_s, nns)
        .+ L2_norm*sum(norm, params(nns))# Add this to your loss
end

"""
    loss_frag(x,ap,b,network,Mo)

fragility should scale with the ration between Cpconf(Tg)/Sconf(Tg) (Webb, 2008, Giordano et Russell, 2017)

We add a loss function to use this constraint in our model
"""
function loss_frag(x,ap,b,network,Mo) # we add this as a constraint
    return mse(fragility(x,network), Mo.*(1.0.+(ap .+ b.*tg(x,network))./ScTg(x,network)))
end
