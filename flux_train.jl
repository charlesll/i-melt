# Copyright Charles Le Losq
"""
    train_nn(path_data,mod_path_out,mod_suffix)

General function to train the network and save it
"""
function train_nn(path_data,mod_path_out,mod_suffix;
                    max_epoch = 5000, nb_neurons = 100, p_drop = 0.3, nb_layers = 3,
                    pretraining=true, verbose = true, figures=false)

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

    # Loading Raman dataset
    X_raman_train = Float32.(h5read("./data/NKAS_DataSet.hdf5","X_raman_train"))
    y_raman_train = Float32.((h5read("./data/NKAS_DataSet.hdf5","y_raman_train")))
    X_raman_valid = Float32.(h5read("./data/NKAS_DataSet.hdf5","X_raman_test"))
    y_raman_valid = Float32.((h5read("./data/NKAS_DataSet.hdf5","y_raman_test")))

    # Loading density dataset
    X_density_train = Float32.(h5read("./data/NKAS_density.hdf5","X_density_train"))
    X_density_valid = Float32.(h5read("./data/NKAS_density.hdf5","X_density_valid"))
    X_density_test = Float32.(h5read("./data/NKAS_density.hdf5","X_density_test"))

    y_density_train = Float32.(h5read("./data/NKAS_density.hdf5","y_density_train"))
    y_density_valid = Float32.(h5read("./data/NKAS_density.hdf5","y_density_valid"))
    y_density_test = Float32.(h5read("./data/NKAS_density.hdf5","y_density_test"))

    #
    # NETWORK DEFINITION
    #

    Ae = param([-2.11]) # Pre-exp Param

    # Network architecture
    nb_channels_raman = size(y_raman_train,1) # number of channels on Raman spectra

    # Hidden Layers
    c1 = Dense(4, nb_neurons, relu)
    c2 = Dense(nb_neurons, nb_neurons, relu)
    c3 = Dense(nb_neurons, nb_neurons, relu)
    c4 = Dense(nb_neurons, nb_neurons, relu)
    c5 = Dense(nb_neurons, nb_neurons, relu)
    c6 = Dense(nb_neurons, nb_neurons, relu)

    # Output Layers
    cout_thermo = Dense(nb_neurons, 4,initb=init_both) #initW = glorot_uniform
    cout_raman =  Dense(nb_neurons,nb_channels_raman)

    # Core : common network
    # Number of layers can be adjust3ed between 1 and 6
    if nb_layers == 1
        core = Chain(c1, Dropout(p_drop))
    elseif nb_layers == 2
        core = Chain(c1, Dropout(p_drop),
            c2, Dropout(p_drop))
    elseif nb_layers == 3
        core = Chain(c1, Dropout(p_drop),
            c2, Dropout(p_drop),
            c3, Dropout(p_drop))
    elseif nb_layers == 4
        core = Chain(c1, Dropout(p_drop),
            c2, Dropout(p_drop),
            c3, Dropout(p_drop),
            c4, Dropout(p_drop))
    elseif nb_layers == 5
        core = Chain(c1, Dropout(p_drop),
            c2, Dropout(p_drop),
            c3, Dropout(p_drop),
            c4, Dropout(p_drop),
            c5, Dropout(p_drop))
    elseif nb_layers == 6
        core = Chain(c1, Dropout(p_drop),
            c2, Dropout(p_drop),
            c3, Dropout(p_drop),
            c4, Dropout(p_drop),
            c5, Dropout(p_drop),
            c6, Dropout(p_drop))
    end

    # Core + outputs
    nnr = Chain(core, cout_raman) |> gpu
    nns = Chain(core, cout_thermo) |> gpu

    # PREPARING DATA / FINAL
    x_train_, y_train_, T_train_, ap_train_, b_train_, sc_train_, tg_train_ = prepare_data(X_train,y_train)
    x_valid_, y_valid_, T_valid_, ap_valid_, b_valid_, sc_valid_, tg_valid_ = prepare_data(X_valid,y_valid)

    x_entro_train_, y_entro_train_, T_entro_train_, ap_entro_train_, b_entro_train_, sc_entro_train_, tg_entro_train_ = prepare_data(X_entropy_train,X_entropy_train[11,:])
    x_entro_valid_, y_entro_valid_, T_entro_valid_, ap_entro_valid_, b_entro_valid_, sc_entro_valid_, tg_entro_valid_ = prepare_data(X_entropy_valid,X_entropy_valid[11,:])

    if verbose == true
        print("\nloss Raman: $(loss_raman(X_raman_train, y_raman_train, nnr))")
        print("\nloss tg: $(loss_tg(x_entro_train_, tg_entro_train_, nns))")
        print("\nloss sc: $(loss_sc(x_entro_train_, sc_entro_train_, nns))")
        print("\nloss density: $(loss_density(X_density_train, y_density_train, nns))")
        print("\nloss ag train: $(loss_n_ag(x_train_, T_train_ ,ap_train_, b_train_, y_train_, nns, Ae))")
        print("\nloss myega train: $(loss_n_am(x_train_, T_train_ , y_train, nns, Ae))")
    end


    #
    # PRETRAINING
    #
    if pretraining == true
        println("\nPretraining activated. Pretraining...")
        #### Raman pre-training followed by entropy/Tg pre-training
        record_loss_raman_train = Float64[]
        record_loss_raman_valid = Float64[]

        testmode!(nns,false)
        testmode!(nnr,false)

        # loop details
        epoch_idx = 1; nb_epoch = 2000
        optimal_epochs_s = 1; early_stop = 1; patience = 100
        min_loss_val = 30000000.0

        p = ProgressMeter.Progress(patience, 1)   # minimum update interval: 1 second
        while early_stop < patience # with realy stopping

            evalcb = () -> (push!(record_loss_raman_train, loss_raman(X_raman_train, y_raman_train,nnr).data),
                            push!(record_loss_raman_valid, loss_raman(X_raman_valid, y_raman_valid,nnr).data))
            Flux.train!(loss_raman, params(nnr), [(X_raman_train,y_raman_train,nnr)], ADAM(0.001), cb = throttle(evalcb, 1))

            # Early stopping criterion
            if record_loss_raman_valid[epoch_idx] < min_loss_val
                early_stop = 1
                optimal_epochs = epoch_idx
                min_loss_val = record_loss_raman_valid[epoch_idx]
            else
                early_stop += 1
            end

            ProgressMeter.update!(p, early_stop)
            epoch_idx += 1
        end

        println(mean(record_loss_raman_train[end-20:end]))
        println(mean(record_loss_raman_valid[end-20:end]))

        if figures == true # just for myself, for quick visualization
            plot(record_loss_raman_train)
            plot!(record_loss_raman_valid)
            savefig("./figures/raman_loss")
        end

        #### Now entropy pre-training

        # record loss
        record_loss_sc_train = Float64[]
        record_loss_sc_valid = Float64[]

        # loop details
        epoch_idx = 1; optimal_epochs_s = 0;
        early_stop = 1; patience = 100
        min_loss_val = 30000000.0

        # loop
        p = ProgressMeter.Progress(patience, 1)   # minimum update interval: 1 second
        while early_stop < patience

            evalcb = () -> (push!(record_loss_sc_train, loss_tg_sc_d(x_entro_train_, tg_entro_train_, sc_entro_train_, X_density_train, y_density_train,nns).data),
                push!(record_loss_sc_valid, loss_tg_sc_d(x_entro_valid_, tg_entro_valid_, sc_entro_valid_,X_density_valid,y_density_valid,nns).data))

            Flux.train!(loss_tg_sc_d, params(nns), [(x_entro_train_, tg_entro_train_, sc_entro_train_,X_density_train, y_density_train,nns)], ADAM(0.001), cb = throttle(evalcb, 1))

            ProgressMeter.update!(p, early_stop)

            # Early stopping criterion
            if record_loss_sc_valid[epoch_idx] < min_loss_val
                early_stop = 0
                optimal_epochs = epoch_idx
                min_loss_val = record_loss_sc_valid[epoch_idx]
            else
                early_stop += 1
            end
            epoch_idx += 1
        end

        println("Entropy-Tg loss")
        println(mean(record_loss_sc_train[end-20:end]))
        println(mean(record_loss_sc_valid[end-20:end]))

        if figures == true # Plot the entropy and Tg comparisons
            # Plot
            plot(record_loss_sc_train,label="train",xlabel="iterations")
            plot!(record_loss_sc_valid,label="valid")
            savefig("./figures/entropy_loss")

            # second plot: comparison between predicted and measurements
            testmode!(nnr)
            testmode!(nns)

            scatter([sc_entro_train_',tg_entro_train_',y_density_train'], # x values
            [ScTg(x_entro_train_,nns).data[:],tg(x_entro_train_,nns).data[:],density(X_density_train,nns).data[:]], # y values
            layout=3, # layout
            label=["Train" "Train" "Train"])

            scatter!([sc_entro_valid_',tg_entro_valid_',y_density_valid'], # x values
            [ScTg(x_entro_valid_,nns).data[:],tg(x_entro_valid_,nns).data[:],density(X_density_valid,nns).data[:]], # y values
            label=["Valid" "Valid" "Valid"])

            savefig("./figures/pretrain_tg_s_d")
        end

    end

    #
    # GLOBAL TRAINING
    #

    # Global loss function, weigth were manually adjusted
    L2_norm = 0.1
    loss_global(x, T, ap, b, y_target, x2, tg2_target, sc2_target, x_raman, y_raman, x_density, y_density, nnr, nns, Ae) =
        100.0.*loss_n_ag(x, T, ap, b, y_target, nns, Ae) .+
        100.0.*loss_n_am(x, T, y_target, nns, Ae) .+
        loss_tg(x2,tg2_target, nns) .+
        100.0.*loss_sc(x2,sc2_target, nns) .+
        1000.0.*loss_density(x_density,y_density, nns) .+
        10.0.*loss_raman(x_raman, y_raman, nnr) .+
        L2_norm*sum(norm, params(nnr,nns))
    # Loss record
    record_loss_train = Float64[]
    record_loss_valid = Float64[]

    # Make sure that dropout is active
    testmode!(nnr,false)
    testmode!(nns,false)

    epoch_idx = 1; optimal_epochs_s = 0;
    early_stop = 1; patience = 20
    min_loss_val = 30000000.0;

    println("Starting the training...")
    p = ProgressMeter.Progress(max_epoch, 1)   # minimum update interval: 1 second

    # Training dataset
    dataset = [(x_train_, T_train_ ,ap_train_, b_train_, y_train_,
                x_entro_train_, tg_entro_train_, sc_entro_train_,
                X_raman_train, y_raman_train, X_density_train, y_density_train, nnr, nns, Ae)]

    while epoch_idx < max_epoch

        evalcb = () -> (push!(record_loss_train, loss_global(x_train_, T_train_ ,ap_train_, b_train_, y_train_,
                    x_entro_train_, tg_entro_train_, sc_entro_train_,X_raman_train, y_raman_train, X_density_train, y_density_train, nnr, nns, Ae).data),
                    push!(record_loss_valid, loss_global(x_valid_, T_valid_ ,ap_valid_, b_valid_, y_valid_,
                    x_entro_valid_, tg_entro_valid_, sc_entro_valid_,X_raman_valid, y_raman_valid, X_density_valid, y_density_valid, nnr, nns, Ae).data))
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

    println("\nTraining done, file saved at",mod_path_out,"with suffix",mod_suffix)
    if  verbose == true
        println("loss Raman: $(loss_raman(X_raman_train, y_raman_train, nnr))")
        println("loss tg: $(loss_tg(x_entro_train_, tg_entro_train_, nns))")
        println("loss sc: $(loss_sc(x_entro_train_, sc_entro_train_, nns))")
        println("loss density: $(loss_density(X_density_train, y_density_train, nns))")
        println("loss n train: $(loss_n_ag(x_train_, T_train_ ,ap_train_, b_train_, y_train_, nns, Ae))")
        println("loss myega train: $(loss_n_am(x_train_, T_train_ , y_train, nns, Ae))")
    end
    println("Global loss:")
    println(mean(record_loss_train[end-5:end]))
    println(mean(record_loss_valid[end-5:end]))

    if figures == true # Plot the entropy and Tg comparisons
        # Plot
        plot(record_loss_train,label="train",xlabel="iterations")
        plot!(record_loss_valid,label="valid")
        savefig("./figures/global_loss")
    end

end
