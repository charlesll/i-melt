# Copyright Charles Le Losq
"""
    train_nn(path_data,mod_path_out,mod_suffix;
             max_epoch = 5000, nb_neurons = 100, p_drop = 0.3, nb_layers = 3,
             pretraining=true, verbose = true, figures=false)

General function to train the network and save it
"""
function train_nn(path_data,mod_path_out,mod_suffix;
                    max_epoch = 5000, nb_neurons = 100, p_drop = 0.3, nb_layers = 3,
                    pretraining=true, verbose = true, figures=false)

        X_columns = h5read(path_data, "X_columns")

    # Entropy dataset
    X_entropy_train = h5read(path_data, "X_entropy_train")
    y_entropy_train = h5read(path_data, "y_entropy_train")

    X_entropy_valid = h5read(path_data, "X_entropy_valid")
    y_entropy_valid = h5read(path_data, "y_entropy_valid")

    X_entropy_test = h5read(path_data, "X_entropy_test")
    y_entropy_test = h5read(path_data, "y_entropy_test")

    # Viscosity dataset
    X_train = h5read(path_data, "X_train")
    y_train = h5read(path_data, "y_train")

    X_valid = h5read(path_data, "X_valid")
    y_valid = h5read(path_data, "y_valid")

    X_test = h5read(path_data, "X_test")
    y_test = h5read(path_data, "y_test")

    # Tg dataset
    X_tg_train = h5read(path_data,"X_tg_train")
    X_tg_valid= h5read(path_data,"X_tg_valid")
    X_tg_test = h5read(path_data,"X_tg_test")

    y_tg_train = h5read(path_data,"y_tg_train")
    y_tg_valid = h5read(path_data,"y_tg_valid")
    y_tg_test = h5read(path_data,"y_tg_test")

    # Raman dataset
    X_raman_train = Float32.(h5read("./data/NKAS_DataSet.hdf5","X_raman_train"))
    y_raman_train = Float32.((h5read("./data/NKAS_DataSet.hdf5","y_raman_train")))
    X_raman_valid = Float32.(h5read("./data/NKAS_DataSet.hdf5","X_raman_test"))
    y_raman_valid = Float32.((h5read("./data/NKAS_DataSet.hdf5","y_raman_test")))

    # Density dataset
    X_density_train = Float32.(h5read("./data/NKAS_density.hdf5","X_density_train"))
    X_density_valid = Float32.(h5read("./data/NKAS_density.hdf5","X_density_valid"))
    X_density_test = Float32.(h5read("./data/NKAS_density.hdf5","X_density_test"))

    y_density_train = Float32.(h5read("./data/NKAS_density.hdf5","y_density_train"))
    y_density_valid = Float32.(h5read("./data/NKAS_density.hdf5","y_density_valid"))
    y_density_test = Float32.(h5read("./data/NKAS_density.hdf5","y_density_test"))

    # Loading fake dataset
    X_FragLoss = Float32.(h5read("./data/X_FragLoss.hdf5","X_gen"))
    ap_FragLoss = ap(X_FragLoss)
    b_FragLoss = b(X_FragLoss)

    if vervose == true
        println("loaded")

        println("\nChemistry order in X arrays is")
        println(X_columns)
        println("\nShape of X train and valid is")
        println(size(X_train))
        println(size(X_valid))

        println("\nSize of Raman datasets")
        println(size(X_raman_train))
        println(size(y_raman_train))
        println(size(X_raman_valid))
        println(size(y_raman_valid))

        println("\nShape of y datasets")
        println(size(y_train))
        println(size(y_entropy_train))
        println(size(y_tg_train))
        println(size(y_density_train))
    end

    #
    # NETWORK DEFINITION
    #

    Ae = param([-2.11]) # Pre-exp Param
    Mo = param([15.0]) # min fragi

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
    x_train_, y_train_, T_train_, ap_train_, b_train_ = prepare_data(X_train,y_train)
    x_valid_, y_valid_, T_valid_, ap_valid_, b_valid_ = prepare_data(X_valid,y_valid)

    if verbose == true
        println("loss Raman: $(loss_raman(X_raman_train, y_raman_train, nnr))")
        println("loss tg: $(loss_tg(X_tg_train, y_tg_train, nns))")
        println("loss sc: $(loss_sc(X_entropy_train, y_entropy_train, nns))")
        println("loss density: $(loss_density(X_density_train, y_density_train, nns))")
        println("loss AG train: $(loss_n_ag(x_train_, T_train_ , ap_train_, b_train_, y_train_, nns, Ae))")
        println("loss MYEGA train: $(loss_n_myega(x_train_, T_train_ ,y_train, nns, Ae))")
        println("loss Frag train: $(loss_frag(x_train_, ap_train_, b_train_, nns, Mo))")
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
        early_stop = 1; patience = 300
        min_loss_val = 30000000.0

        # loop
        p = ProgressMeter.Progress(patience, 1)   # minimum update interval: 1 second
        while early_stop < patience

            evalcb = () -> (push!(record_loss_sc_train, loss_tg_d_sc(X_tg_train, y_tg_train,
                                                                    X_density_train, y_density_train,
                                                                    X_entropy_train, y_entropy_train,
                                                                    nns).data),
                push!(record_loss_sc_valid, loss_tg_d_sc(X_tg_valid, y_tg_valid,
                                                        X_density_valid,y_density_valid,
                                                        X_entropy_valid, y_entropy_valid,
                                                        nns).data))
            dataset = [(X_tg_train, y_tg_train,X_density_train, y_density_train,X_entropy_train, y_entropy_train,nns)]
            Flux.train!(loss_tg_d_sc, params(nns), dataset, ADAM(0.001), cb = throttle(evalcb, 1))

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

        println("Tg-density-entropy loss")
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

            scatter([y_tg_train'[:],y_density_train'[:],y_entropy_train'[:]], # x values
            [tg(X_tg_train,nns).data'[:],density(X_density_train,nns).data'[:],ScTg(X_entropy_train,nns).data'[:]], # y values
            layout=3, # layout
            label=["Train" "Train" "Train"])

            scatter!([y_tg_valid'[:],y_density_valid'[:],y_entropy_valid'[:]], # x values
            [tg(X_tg_valid,nns).data'[:],density(X_density_valid,nns).data'[:],ScTg(X_entropy_valid,nns).data'[:]], # y values
            label=["Valid" "Valid" "Valid"])

            savefig("./figures/pretrain_tg_s_d")
        end

    end

    #
    # GLOBAL TRAINING
    #

    println("\nPre-training done, errors are of:")
    if  verbose == true
        println("loss Raman: $(loss_raman(X_raman_train, y_raman_train, nnr))")
        println("loss tg: $(loss_tg(X_tg_train, y_tg_train, nns))")
        println("loss sc: $(loss_sc(X_entropy_train, y_entropy_train, nns))")
        println("loss density: $(loss_density(X_density_train, y_density_train, nns))")
        println("loss AG train: $(loss_n_ag(x_train_, T_train_ , ap_train_, b_train_, y_train_, nns, Ae))")
        println("loss MYEGA train: $(loss_n_myega(x_train_, T_train_ , y_train, nns, Ae))")
        println("loss Frag train: $(loss_frag(X_FragLoss, ap_FragLoss, b_FragLoss, nns, Mo))")
    end

    # Global loss function, weigth were manually adjusted
    L2_norm = 0.1
    loss_global(x, T, ap, b, y_target, x_tg, y_tg, x_raman, y_raman, x_density, y_density, x_sc, y_sc, x_frag, ap_frag, b_frag, nnr, nns, Ae, Mo) =
        200.0.*loss_n_ag(x, T, ap, b, y_target, nns, Ae) .+ # more important
        100.0.*loss_n_myega(x, T, y_target, nns, Ae) .+ # less important
        loss_tg(x_tg,y_tg, nns) .+ # Tg loss, always important
        100.0.*loss_sc(x_sc, y_sc, nns) .+ # entropy, not very, just as a loose constraint
        1000.0.*loss_density(x_density,y_density, nns) .+ # density, important
        10.0.*loss_raman(x_raman, y_raman, nnr) .+ # Raman is a loose constraint too
        50.0 .*loss_frag(x_frag, ap_frag, b_frag, nns, Mo) .+ # Minimizing the different between viscous fragility and its thermodynamic calculation
        L2_norm*sum(norm, params(nnr,nns)) # avoiding overfitting by adding a L2_norm; forces weights toward 0
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
                X_tg_train, y_tg_train,
                X_raman_train, y_raman_train,
                X_density_train, y_density_train,
                X_entropy_train, y_entropy_train,
                X_FragLoss, ap_FragLoss, b_FragLoss,
                nnr, nns, Ae, Mo)]

    while epoch_idx < max_epoch

        evalcb = () -> (push!(record_loss_train, loss_global(x_train_, T_train_ ,ap_train_, b_train_, y_train_,
                    X_tg_train, y_tg_train,X_raman_train, y_raman_train, X_density_train, y_density_train,
                    X_entropy_train, y_entropy_train,X_FragLoss, ap_FragLoss, b_FragLoss,
                     nnr, nns, Ae, Mo).data),
                    push!(record_loss_valid, loss_global(x_valid_, T_valid_ ,ap_valid_, b_valid_, y_valid_,
                    X_tg_valid, y_tg_valid,X_raman_valid, y_raman_valid, X_density_valid, y_density_valid,
                    X_entropy_valid, y_entropy_valid,X_FragLoss, ap_FragLoss, b_FragLoss,
                    nnr, nns, Ae, Mo).data))
        Flux.train!(loss_global, params(Ae,Mo,nnr,nns), dataset, ADAM(0.001), cb = throttle(evalcb, 1))

        ProgressMeter.update!(p, epoch_idx)

        # Early stopping criterion / we don't use it except for saving
        if record_loss_valid[epoch_idx] < min_loss_val
            early_stop = 0
            optimal_epochs = epoch_idx
            min_loss_val = record_loss_valid[epoch_idx]
            @save mod_path_out*"nns"*mod_suffix*".bson" nns
            @save mod_path_out*"nnr"*mod_suffix*".bson" nnr
            @save mod_path_out*"Ae"*mod_suffix*".bson" Ae
            @save mod_path_out*"Mo"*mod_suffix*".bson" Mo
        else
            early_stop += 1
        end
        epoch_idx += 1

    end

    println("\nTraining done, file saved at",mod_path_out,"with suffix",mod_suffix)
    if  verbose == true
        println("loss Raman: $(loss_raman(X_raman_train, y_raman_train, nnr))")
        println("loss tg: $(loss_tg(X_tg_train, y_tg_train, nns))")
        println("loss sc: $(loss_sc(X_entropy_train, y_entropy_train, nns))")
        println("loss density: $(loss_density(X_density_train, y_density_train, nns))")
        println("loss AG train: $(loss_n_ag(x_train_, T_train_ , ap_train_, b_train_, y_train_, nns, Ae))")
        println("loss MYEGA train: $(loss_n_myega(x_train_, T_train_ , y_train, nns, Ae))")
        println("loss Frag train: $(loss_frag(X_FragLoss, ap_FragLoss, b_FragLoss, nns, Mo))")
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
