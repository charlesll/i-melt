# General function to train the network and save it
function train_nn(path_data,mod_path_out,mod_suffix)
    X_columns = h5read(path_data, "X_columns")

    X_entropy_train = h5read(path_data, "X_entropy_train")
    X_entropy_train_sc = h5read(path_data, "X_entropy_train_sc")

    X_entropy_valid = h5read(path_data, "X_entropy_valid")
    X_entropy_valid_sc = h5read(path_data, "X_entropy_valid_sc")

    X_entropy_test = h5read(path_data, "X_entropy_test")
    X_entropy_test_sc = h5read(path_data, "X_entropy_test_sc")

    X_entropy_tv = [X_entropy_train X_entropy_valid]
    X_entropy_sc_tv = [X_entropy_train_sc X_entropy_valid_sc]

    X_tv = h5read(path_data, "X_tv")
    X_tv_sc = h5read(path_data, "X_tv_sc")
    y_tv = h5read(path_data, "y_tv")

    X_train = h5read(path_data, "X_train")
    X_train_sc = h5read(path_data, "X_train_sc")
    y_train = h5read(path_data, "y_train")

    X_valid = h5read(path_data, "X_valid")
    X_valid_sc = h5read(path_data, "X_valid_sc")
    y_valid = h5read(path_data, "y_valid")

    X_test = h5read(path_data, "X_test")
    X_test_sc = h5read(path_data, "X_test_sc")
    y_test = h5read(path_data, "y_test")

    X_scaler_mean = h5read(path_data, "X_scaler_mean")
    X_scaler_var = h5read(path_data, "X_scaler_var")

    X_raman_train = Float32.(h5read("./data/NKAS_DataSet.hdf5","X_raman_train"))
    y_raman_train = Float32.((h5read("./data/NKAS_DataSet.hdf5","y_raman_train")))
    X_raman_valid = Float32.(h5read("./data/NKAS_DataSet.hdf5","X_raman_test"))
    y_raman_valid = Float32.((h5read("./data/NKAS_DataSet.hdf5","y_raman_test")))

    println("loaded")
    println("\nFeatures in X_ arrays are")
    println(X_columns)
    println("\nShape of X train and valid is")
    println(size(X_train))
    println(size(X_valid))

    print("Size of Raman datasets")
    print(size(X_raman_train))
    print(size(y_raman_train))
    print(size(X_raman_valid))
    print(size(y_raman_valid))

    #
    # NETWORK DEFINITION
    #

    Ae = param([-2.11]) # Pre-exp Param

    # Network architecture
    nb_neurons = 100
    p_drop = 0.3

    nb_channels_raman = size(y_raman_train,1) # number of channels on Raman spectra

    # Hidden Layers
    c1 = Dense(4, nb_neurons, relu)
    c2 = Dense(nb_neurons, nb_neurons, relu)
    c3 = Dense(nb_neurons, nb_neurons, relu)

    # Output Layers
    cout_thermo = Dense(nb_neurons, 3,initb=init_both) #initW = glorot_uniform
    cout_raman =  Dense(nb_neurons,nb_channels_raman)

    # Core : common network
    core = Chain(c1, Dropout(p_drop),
        c2, Dropout(p_drop),
        c3, Dropout(p_drop))

    # Core + outputs
    nnr = Chain(core, cout_raman) |> gpu
    nns = Chain(core, cout_thermo) |> gpu

    # PREPARING DATA / FINAL
    x_train_, y_train_, T_train_, ap_train_, b_train_, sc_train_, tg_train_ = prepare_datas(X_train,y_train)
    x_valid_, y_valid_, T_valid_, ap_valid_, b_valid_, sc_valid_, tg_valid_ = prepare_datas(X_valid,y_valid)

    x_entro_train_, y_entro_train_, T_entro_train_, ap_entro_train_, b_entro_train_, sc_entro_train_, tg_entro_train_ = prepare_datas(X_entropy_train,X_entropy_train[11,:])
    x_entro_valid_, y_entro_valid_, T_entro_valid_, ap_entro_valid_, b_entro_valid_, sc_entro_valid_, tg_entro_valid_ = prepare_datas(X_entropy_valid,X_entropy_valid[11,:])

    print("\nloss Raman: $(loss_raman(X_raman_train, y_raman_train, nnr))")
    print("\nloss tg: $(loss_tg(x_entro_train_, tg_entro_train_, nns))")
    print("\nloss sc: $(loss_sc(x_entro_train_, sc_entro_train_, nns))")
    print("\nloss n train: $(loss_n(x_train_, T_train_ ,ap_train_, b_train_, y_train_, nns, Ae))")
    print("\nloss myega train: $(loss_n_myega(x_train_, T_train_ , y_train, nns, Ae))")

    # Global loss function, weigth were manually adjusted
    L2_norm = 0.1
    loss_global(x, T, ap, b, y_target, x2, tg2_target, sc2_target, x_raman, y_raman, nnr, nns, Ae) =
        loss_n(x, T, ap, b, y_target, nns, Ae) .+
        loss_n_myega(x,T,y_target, nns, Ae) .+
        0.1.*loss_tg(x2,tg2_target, nns) .+
        loss_sc(x2,sc2_target, nns) .+
        0.01.*loss_raman(x_raman, y_raman, nnr) .+
        L2_norm*sum(norm, params(nnr,nns))

    # Loss record
    record_loss_train = Float64[]
    record_loss_valid = Float64[]

    # Make sure that dropout is active
    testmode!(nnr,false)
    testmode!(nns,false)

    epoch_idx = 1; optimal_epochs_s = 0;
    early_stop = 1; patience = 20
    min_loss_val = 30000000.0; max_epoch = 5000

    print("\nStarting the training...")
    p = ProgressMeter.Progress(max_epoch, 1)   # minimum update interval: 1 second

    # Training dataset
    dataset = [(x_train_, T_train_ ,ap_train_, b_train_, y_train_,
                x_entro_train_, tg_entro_train_, sc_entro_train_,
                X_raman_train, y_raman_train, nnr, nns, Ae)]

    while epoch_idx < max_epoch

        evalcb = () -> (push!(record_loss_train, loss_global(x_train_, T_train_ ,ap_train_, b_train_, y_train_,
                    x_entro_train_, tg_entro_train_, sc_entro_train_,X_raman_train, y_raman_train, nnr, nns, Ae).data),
                    push!(record_loss_valid, loss_global(x_valid_, T_valid_ ,ap_valid_, b_valid_, y_valid_,
                    x_entro_valid_, tg_entro_valid_, sc_entro_valid_,X_raman_valid, y_raman_valid, nnr, nns, Ae).data))
        Flux.train!(loss_global, params(Ae,nnr,nns), dataset, ADAM(0.001), cb = throttle(evalcb, 1))

        ProgressMeter.update!(p, epoch_idx)

        # Early stopping criterion / we don't use it except for saving
        if record_loss_valid[epoch_idx] < min_loss_val
            early_stop = 0
            optimal_epochs = epoch_idx
            min_loss_val = record_loss_valid[epoch_idx]
            @save mod_path_out*"nns"*mod_suffix*".bson" nns
            @save mod_path_out*"nnr"*mod_suffix*".bson" nnr
            @save mod_path_out*"Ae"*mod_suffix*".bson" Ae
        else
            early_stop += 1
        end
        epoch_idx += 1

    end

    println("Training done, file saved at",mod_path_out,"with suffix",mod_suffix)
    println("loss Raman: $(loss_raman(X_raman_train, y_raman_train, nnr))")
    println("loss tg: $(loss_tg(x_entro_train_, tg_entro_train_, nns))")
    println("loss sc: $(loss_sc(x_entro_train_, sc_entro_train_, nns))")
    println("loss n train: $(loss_n(x_train_, T_train_ ,ap_train_, b_train_, y_train_, nns, Ae))")
    println("loss myega train: $(loss_n_myega(x_train_, T_train_ , y_train, nns, Ae))")
end
