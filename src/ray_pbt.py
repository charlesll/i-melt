import imelt, torch, os

from sklearn.metrics import mean_squared_error

import ray
from ray import tune, air
from ray.air import session, Checkpoint
from ray.tune.examples.mnist_pytorch import ConvNet, get_data_loaders, train, test
from ray.tune.schedulers import PopulationBasedTraining
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Calculation will be performed on {}".format(device))
cluster = True

if cluster == True:
    path = "/gpfs/users/lelosq/neuravi/"
else:
    path = "/home/charles/Sync_me/neuravi_DEV/"

def train(neuralmodel, ds, criterion, optimizer, device='cuda'):
    """train neuralmodel given a dataset, criterion and optimizer

    Parameters
    ----------
    neuralmodel : model
        a neuravi model
    ds : dataset
        dataset from data_loader()
    criterion : pytorch criterion
        the criterion for goodness of fit
    optimizer : pytorch optimizer
        the optimizer to use
    

    Options
    -------
    nb_folds : int, default = 1
        the number of folds for the K-fold training
    device : string, default = "cuda"
        the device where the calculations are made during training

    """
    # set model to train mode
    neuralmodel.train()

    # data tensors are sent to device
    x_visco_train = ds.x_visco_train.to(device)
    y_visco_train = ds.y_visco_train.to(device)
    T_visco_train = ds.T_visco_train.to(device)

    x_raman_train = ds.x_raman_train.to(device)
    y_raman_train = ds.y_raman_train.to(device)

    x_density_train = ds.x_density_train.to(device)
    y_density_train = ds.y_density_train.to(device)

    x_entro_train = ds.x_entro_train.to(device)
    y_entro_train = ds.y_entro_train.to(device)

    x_ri_train = ds.x_ri_train.to(device)
    y_ri_train = ds.y_ri_train.to(device)
    lbd_ri_train = ds.lbd_ri_train.to(device)

    x_cpl_train = ds.x_cpl_train.to(device)
    y_cpl_train = ds.y_cpl_train.to(device)
    T_cpl_train = ds.T_cpl_train.to(device)

    x_elastic_train = ds.x_elastic_train.to(device)
    y_elastic_train = ds.y_elastic_train.to(device)

    x_cte_train = ds.x_cte_train.to(device)
    y_cte_train = ds.y_cte_train.to(device)

    x_abbe_train = ds.x_abbe_train.to(device)
    y_abbe_train = ds.y_abbe_train.to(device)

    x_liquidus_train = ds.x_liquidus_train.to(device)
    y_liquidus_train = ds.y_liquidus_train.to(device)

    # Forward pass on training set
    y_ag_pred_train = neuralmodel.ag(x_visco_train,T_visco_train)
    y_myega_pred_train = neuralmodel.myega(x_visco_train,T_visco_train)
    y_am_pred_train = neuralmodel.am(x_visco_train,T_visco_train)
    y_cg_pred_train = neuralmodel.cg(x_visco_train,T_visco_train)
    y_tvf_pred_train = neuralmodel.tvf(x_visco_train,T_visco_train)
    y_raman_pred_train = neuralmodel.raman_pred(x_raman_train)
    y_density_pred_train = neuralmodel.density_glass(x_density_train)
    y_entro_pred_train = neuralmodel.sctg(x_entro_train)
    y_ri_pred_train = neuralmodel.sellmeier(x_ri_train,lbd_ri_train)
    y_cpl_pred_train = neuralmodel.cpl(x_cpl_train, T_cpl_train)
    y_elastic_pred_train = neuralmodel.elastic_modulus(x_elastic_train)
    y_cte_pred_train = neuralmodel.cte(x_cte_train)
    y_abbe_pred_train = neuralmodel.abbe(x_abbe_train)
    y_liquidus_pred_train = neuralmodel.liquidus(x_liquidus_train)

    # initialise gradient
    optimizer.zero_grad() 

    # Get precisions
    precision_visco = torch.exp(-neuralmodel.log_vars[0])
    precision_raman = torch.exp(-neuralmodel.log_vars[1])
    precision_density = torch.exp(-neuralmodel.log_vars[2])
    precision_entro = torch.exp(-neuralmodel.log_vars[3])
    precision_ri = torch.exp(-neuralmodel.log_vars[4])
    precision_cpl = torch.exp(-neuralmodel.log_vars[5])
    precision_elastic = torch.exp(-neuralmodel.log_vars[6])
    precision_cte = torch.exp(-neuralmodel.log_vars[7])
    precision_abbe = torch.exp(-neuralmodel.log_vars[8])
    precision_liquidus = torch.exp(-neuralmodel.log_vars[9])
        
    # Compute Loss
    loss_ag = precision_visco * criterion(y_ag_pred_train, y_visco_train)
    loss_myega = precision_visco * criterion(y_myega_pred_train, y_visco_train)
    loss_am = precision_visco * criterion(y_am_pred_train, y_visco_train)
    loss_cg = precision_visco * criterion(y_cg_pred_train, y_visco_train)
    loss_tvf = precision_visco * criterion(y_tvf_pred_train, y_visco_train)
    loss_raman = precision_raman * criterion(y_raman_pred_train,y_raman_train)
    loss_density = precision_density * criterion(y_density_pred_train,y_density_train)
    loss_entro = precision_entro * criterion(y_entro_pred_train,y_entro_train)
    loss_ri = precision_ri * criterion(y_ri_pred_train,y_ri_train)
    loss_cpl = precision_cpl * criterion(y_cpl_pred_train,y_cpl_train) 
    loss_elastic = precision_elastic * criterion(y_elastic_pred_train,y_elastic_train)
    loss_cte = precision_cte * criterion(y_cte_pred_train,y_cte_train)
    loss_abbe = precision_abbe * criterion(y_abbe_pred_train,y_abbe_train)
    loss_liquidus = precision_liquidus * criterion(y_liquidus_pred_train,y_liquidus_train)

    loss_train = (loss_ag + loss_myega + loss_am + loss_cg + loss_tvf
                + loss_raman + loss_density + loss_entro + loss_ri 
                + loss_cpl + loss_elastic + loss_cte + loss_abbe + loss_liquidus
                + neuralmodel.log_vars[0] + neuralmodel.log_vars[1] + neuralmodel.log_vars[2] 
                + neuralmodel.log_vars[3] + neuralmodel.log_vars[4] + neuralmodel.log_vars[5] 
                + neuralmodel.log_vars[6] + neuralmodel.log_vars[7] + neuralmodel.log_vars[8]
                + neuralmodel.log_vars[9])

    loss_train.backward() # backward gradient determination
    optimizer.step() # optimiser call and step
    
    return loss_train.item()

def valid(neuralmodel, ds, criterion, device='cuda'):

    # Set model to evaluation mode
    neuralmodel.eval()

    # MONITORING VALIDATION SUBSET
    with torch.no_grad():

        # # Precisions
        precision_visco = torch.exp(-neuralmodel.log_vars[0])
        precision_raman = torch.exp(-neuralmodel.log_vars[1])
        precision_density = torch.exp(-neuralmodel.log_vars[2])
        precision_entro = torch.exp(-neuralmodel.log_vars[3])
        precision_ri = torch.exp(-neuralmodel.log_vars[4])
        precision_cpl = torch.exp(-neuralmodel.log_vars[5])
        precision_elastic = torch.exp(-neuralmodel.log_vars[6])
        precision_cte = torch.exp(-neuralmodel.log_vars[7])
        precision_abbe = torch.exp(-neuralmodel.log_vars[8])
        precision_liquidus = torch.exp(-neuralmodel.log_vars[9])

        # on validation set
        y_ag_pred_valid = neuralmodel.ag(ds.x_visco_valid.to(device),ds.T_visco_valid.to(device))
        y_myega_pred_valid = neuralmodel.myega(ds.x_visco_valid.to(device),ds.T_visco_valid.to(device))
        y_am_pred_valid = neuralmodel.am(ds.x_visco_valid.to(device),ds.T_visco_valid.to(device))
        y_cg_pred_valid = neuralmodel.cg(ds.x_visco_valid.to(device),ds.T_visco_valid.to(device))
        y_tvf_pred_valid = neuralmodel.tvf(ds.x_visco_valid.to(device),ds.T_visco_valid.to(device))
        y_raman_pred_valid = neuralmodel.raman_pred(ds.x_raman_valid.to(device))
        y_density_pred_valid = neuralmodel.density_glass(ds.x_density_valid.to(device))
        y_entro_pred_valid = neuralmodel.sctg(ds.x_entro_valid.to(device))
        y_ri_pred_valid = neuralmodel.sellmeier(ds.x_ri_valid.to(device), ds.lbd_ri_valid.to(device))
        y_clp_pred_valid = neuralmodel.cpl(ds.x_cpl_valid.to(device), ds.T_cpl_valid.to(device))
        y_elastic_pred_valid = neuralmodel.elastic_modulus(ds.x_elastic_valid.to(device))
        y_cte_pred_valid = neuralmodel.cte(ds.x_cte_valid.to(device))
        y_abbe_pred_valid = neuralmodel.abbe(ds.x_abbe_valid.to(device))
        y_liquidus_pred_valid = neuralmodel.liquidus(ds.x_liquidus_valid.to(device))

        # validation loss
        loss_ag_v = precision_visco * criterion(y_ag_pred_valid, ds.y_visco_valid.to(device))
        loss_myega_v = precision_visco * criterion(y_myega_pred_valid, ds.y_visco_valid.to(device))
        loss_am_v = precision_visco * criterion(y_am_pred_valid, ds.y_visco_valid.to(device))
        loss_cg_v = precision_visco * criterion(y_cg_pred_valid, ds.y_visco_valid.to(device))
        loss_tvf_v = precision_visco * criterion(y_tvf_pred_valid, ds.y_visco_valid.to(device))
        loss_raman_v = precision_raman * criterion(y_raman_pred_valid,ds.y_raman_valid.to(device))
        loss_density_v = precision_density * criterion(y_density_pred_valid,ds.y_density_valid.to(device))
        loss_entro_v = precision_entro * criterion(y_entro_pred_valid,ds.y_entro_valid.to(device))
        loss_ri_v = precision_ri * criterion(y_ri_pred_valid,ds.y_ri_valid.to(device))
        loss_cpl_v = precision_cpl * criterion(y_clp_pred_valid,ds.y_cpl_valid.to(device))
        loss_elastic_v = precision_elastic * criterion(y_elastic_pred_valid,ds.y_elastic_valid.to(device))
        loss_cte_v = precision_cte * criterion(y_cte_pred_valid,ds.y_cte_valid.to(device))
        loss_abbe_v = precision_abbe * criterion(y_abbe_pred_valid,ds.y_abbe_valid.to(device))
        loss_liquidus_v = precision_liquidus * criterion(y_liquidus_pred_valid,ds.y_liquidus_valid.to(device))

        loss_v = (loss_ag_v + loss_myega_v + loss_am_v + loss_cg_v + loss_tvf_v
                    + loss_raman_v + loss_density_v + loss_entro_v + loss_ri_v 
                    + loss_cpl_v + loss_elastic_v + loss_cte_v + loss_abbe_v + loss_liquidus_v
                    + neuralmodel.log_vars[0] + neuralmodel.log_vars[1] + neuralmodel.log_vars[2] 
                    + neuralmodel.log_vars[3] + neuralmodel.log_vars[4] + neuralmodel.log_vars[5] 
                    + neuralmodel.log_vars[6] + neuralmodel.log_vars[7] + neuralmodel.log_vars[8]
                    + neuralmodel.log_vars[9])

    return loss_v.item()

def objective(config):
    # counting step and patience
    step = 1
    patience = 0
    min_delta = 0.05

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # set dtype
    dtype = torch.float

    # custom data loader
    ds = imelt.data_loader(path_viscosity = path+"data/NKCMAS_viscosity.hdf5", 
                            path_raman = path+"data/NKCMAS_Raman.hdf5", 
                            path_density = path+"data/NKCMAS_density.hdf5", 
                            path_ri = path+"data/NKCMAS_optical.hdf5", 
                            path_cp = path+"data/NKCMAS_cp.hdf5",
                            path_elastic = path+"data/NKCMAS_em.hdf5",
                            path_cte = path+"data/NKCMAS_cte.hdf5",
                            path_abbe = path+"data/NKCMAS_abbe.hdf5",
                            path_liquidus = path+"data/NKCMAS_tl.hdf5")

    # model declaration
    net = imelt.model(ds.x_visco_train.shape[1],
                        500,
                        4,
                        ds.nb_channels_raman, 
                        config.get("dropout",0.1), 
                        activation_function = torch.nn.GELU()) # declaring model
    
    net.output_bias_init() # we initialize the output bias
    net.to(dtype=dtype, device=device) # set dtype and send to device

    # Define loss and optimizer
    criterion = torch.nn.MSELoss() # the criterion : MSE
    criterion.to(device)

    optimizer = torch.optim.Adam(net.parameters(), 
                                 lr=config.get("lr",0.0003), # 0.0003 is used if no lr is specified
                                 weight_decay=config.get("wd",0.00)) # 0 if not specified

    # If `session.get_checkpoint()` is not None, then we are resuming from a checkpoint.
    if session.get_checkpoint():
        # Load model state and iteration step from checkpoint.
        checkpoint_dict = session.get_checkpoint().to_dict()
        net.load_state_dict(checkpoint_dict["model_state_dict"])
        # Load optimizer state (needed since we're using momentum),
        # then set the `lr` and `momentum` according to the config.
        optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
        for param_group in optimizer.param_groups:
            if "lr" in config:
                param_group["lr"] = config["lr"]
            if "wd" in config:
                param_group["wd"] = config["wd"]
            if "dropout" in config:
                param_group["dropout"] = config["dropout"]

        # Note: Make sure to increment the checkpointed step by 1 to get the current step.
        last_step = checkpoint_dict["step"]
        best_loss_v = checkpoint_dict["best_loss_v"]
        step = last_step + 1

    while True:
        # train and valid
        train_loss = train(net, ds, criterion, optimizer, device=device)
        valid_loss = valid(net, ds, criterion, device=device)

        diff_loss = np.sqrt((train_loss - valid_loss)**2)

        tot_loss = valid_loss + diff_loss
        
        # calculating ES criterion
        # calculate patience term to trigger early stopping
        if step == 1:
            best_loss_v = valid_loss
        elif valid_loss <= (best_loss_v - min_delta): # if improvement is significant, this saves the model
            patience = 0
        else:
            patience += 1

        checkpoint = None
        if step % config["checkpoint_interval"] == 0:
            # Every `checkpoint_interval` steps, checkpoint our current state.
            checkpoint = Checkpoint.from_dict({
                "step": step,
                "best_loss_v": best_loss_v,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            })

        session.report(
            {"valid_loss": valid_loss, "tot_loss": tot_loss, "patience": patience, "step": step},
            checkpoint=checkpoint
        )
        step += 1

########################   
# ARE WE RUNNING A TEST?
########################
smoke_test = False
########################

# 2. define perturbation intervals
perturbation_interval = 1 if smoke_test else 400
scheduler = PopulationBasedTraining(
    time_attr="step",
    perturbation_interval=perturbation_interval,
    metric="valid_loss",
    mode="min",
    burn_in_period= 0 if smoke_test else 1000,
    hyperparam_mutations={
        # distribution for resampling
        "lr": tune.uniform(1e-5, 5e-4),
        "wd": tune.uniform(1e-7, 1e-3),
        # allow perturbations within this set of categorical values
        "dropout": tune.uniform(0.13, 0.30)
    },
)

# 3. Define a search space and initialize the search algorithm.
search_space = {"lr": tune.uniform(5e-5, 5e-4), 
                "wd": tune.loguniform(1e-7, 1e-2), 
                "dropout": tune.uniform(0.1, 0.3),
                "checkpoint_interval": perturbation_interval}

#4. Start a Tune run that maximizes mean accuracy and stops after 5 iterations.
if ray.is_initialized():
    ray.shutdown()
ray.init()

tuner = tune.Tuner(
    tune.with_resources(objective, {"gpu": 2 if smoke_test else 4}),
    run_config=air.RunConfig(
        name="pbt_test",
        # Stop when patience or training iterations exceeds the specified values.
        stop={"step": 5 if smoke_test else 4500},#{"patience": 250},
        verbose=1,
        checkpoint_config=air.CheckpointConfig(
            checkpoint_score_attribute="valid_loss",
            num_to_keep= 2 if smoke_test else 4,
        ),
        #local_dir="/tmp/ray_results",
    ),
    tune_config=tune.TuneConfig(
        scheduler=scheduler,
        num_samples= 2 if smoke_test else 4,
    ),
    param_space=search_space,
)

results_grid = tuner.fit()
#print("Best config is:", results.get_best_result(metric="mean_loss", mode="min").config)