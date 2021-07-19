import numpy as np
import torch, time
import h5py

class data_loader():
    """custom data loader for batch training

    """
    def __init__(self,path_viscosity,path_raman,path_density, path_ri, device, scaling = False):
        """
        Inputs
        ------
        path_viscosity : string
            path for the viscosity HDF5 dataset

        path_raman : string
            path for the Raman spectra HDF5 dataset

        path_density : string
            path for the density HDF5 dataset

        path_ri : String
            path for the refractive index HDF5 dataset

        device : CUDA
        
        scaling : False or True
            Scales the input chemical composition. 
            WARNING : Does not work currently as this is a relic of testing this effect, 
            but we chose not to scale inputs and Cp are calculated with unscaled values in the network."""
        
        f = h5py.File(path_viscosity, 'r')

        # List all groups
        self.X_columns = f['X_columns'][()]

        # Entropy dataset
        X_entropy_train = f["X_entropy_train"][()]
        y_entropy_train = f["y_entropy_train"][()]

        X_entropy_valid = f["X_entropy_valid"][()]
        y_entropy_valid = f["y_entropy_valid"][()]

        X_entropy_test = f["X_entropy_test"][()]
        y_entropy_test = f["y_entropy_test"][()]

        # Viscosity dataset
        X_train = f["X_train"][()]
        y_train = f["y_train"][()]

        X_valid = f["X_valid"][()]
        y_valid = f["y_valid"][()]

        X_test = f["X_test"][()]
        y_test = f["y_test"][()]

        # Tg dataset
        X_tg_train = f["X_tg_train"][()]
        X_tg_valid= f["X_tg_valid"][()]
        X_tg_test = f["X_tg_test"][()]

        y_tg_train = f["y_tg_train"][()]
        y_tg_valid = f["y_tg_valid"][()]
        y_tg_test = f["y_tg_test"][()]

        f.close()

        # Raman dataset
        f = h5py.File(path_raman, 'r')
        X_raman_train = f["X_raman_train"][()]
        y_raman_train = f["y_raman_train"][()]
        X_raman_valid = f["X_raman_test"][()]
        y_raman_valid = f["y_raman_test"][()]
        f.close()

        # Density dataset
        f = h5py.File(path_density, 'r')
        X_density_train = f["X_density_train"][()]
        X_density_valid = f["X_density_valid"][()]
        X_density_test = f["X_density_test"][()]

        y_density_train = f["y_density_train"][()]
        y_density_valid = f["y_density_valid"][()]
        y_density_test = f["y_density_test"][()]
        f.close()

        # Refractive Index (ri) dataset
        f = h5py.File(path_ri, 'r')
        X_ri_train = f["X_ri_train"][()]
        X_ri_valid = f["X_ri_valid"][()]
        X_ri_test = f["X_ri_test"][()]

        lbd_ri_train = f["lbd_ri_train"][()]
        lbd_ri_valid = f["lbd_ri_valid"][()]
        lbd_ri_test = f["lbd_ri_test"][()]

        y_ri_train = f["y_ri_train"][()]
        y_ri_valid = f["y_ri_valid"][()]
        y_ri_test = f["y_ri_test"][()]
        f.close()

        # grabbing number of Raman channels
        self.nb_channels_raman = y_raman_valid.shape[1]

        # preparing data for pytorch

        # Scaler
        # Warning : this was done for tests and currently will not work, 
        # as Cp are calculated from unscaled mole fractions...
        if scaling ==  True:
            X_scaler_mean = np.mean(X_train[:,0:4], axis=0)
            X_scaler_std = np.std(X_train[:,0:4], axis=0)
        else:
            X_scaler_mean = 0.0
            X_scaler_std = 1.0
            
        # The following lines perform scaling (not needed, not active), 
        # put the data in torch tensors and send them to device (GPU or CPU, as requested)
        
        # viscosity
        self.x_visco_train = torch.FloatTensor(self.scaling(X_train[:,0:4],X_scaler_mean,X_scaler_std)).to(device)
        self.T_visco_train = torch.FloatTensor(X_train[:,4].reshape(-1,1)).to(device)
        self.y_visco_train = torch.FloatTensor(y_train[:,0].reshape(-1,1)).to(device)

        self.x_visco_valid = torch.FloatTensor(self.scaling(X_valid[:,0:4],X_scaler_mean,X_scaler_std)).to(device)
        self.T_visco_valid = torch.FloatTensor(X_valid[:,4].reshape(-1,1)).to(device)
        self.y_visco_valid = torch.FloatTensor(y_valid[:,0].reshape(-1,1)).to(device)

        self.x_visco_test = torch.FloatTensor(self.scaling(X_test[:,0:4],X_scaler_mean,X_scaler_std)).to(device)
        self.T_visco_test = torch.FloatTensor(X_test[:,4].reshape(-1,1)).to(device)
        self.y_visco_test = torch.FloatTensor(y_test[:,0].reshape(-1,1)).to(device)

        # entropy
        self.x_entro_train = torch.FloatTensor(self.scaling(X_entropy_train[:,0:4],X_scaler_mean,X_scaler_std)).to(device)
        self.y_entro_train = torch.FloatTensor(y_entropy_train[:,0].reshape(-1,1)).to(device)

        self.x_entro_valid = torch.FloatTensor(self.scaling(X_entropy_valid[:,0:4],X_scaler_mean,X_scaler_std)).to(device)
        self.y_entro_valid = torch.FloatTensor(y_entropy_valid[:,0].reshape(-1,1)).to(device)

        self.x_entro_test = torch.FloatTensor(self.scaling(X_entropy_test[:,0:4],X_scaler_mean,X_scaler_std)).to(device)
        self.y_entro_test = torch.FloatTensor(y_entropy_test[:,0].reshape(-1,1)).to(device)

        # tg
        self.x_tg_train = torch.FloatTensor(self.scaling(X_tg_train[:,0:4],X_scaler_mean,X_scaler_std)).to(device)
        self.y_tg_train = torch.FloatTensor(y_tg_train.reshape(-1,1)).to(device)

        self.x_tg_valid = torch.FloatTensor(self.scaling(X_tg_valid[:,0:4],X_scaler_mean,X_scaler_std)).to(device)
        self.y_tg_valid = torch.FloatTensor(y_tg_valid.reshape(-1,1)).to(device)

        self.x_tg_test = torch.FloatTensor(self.scaling(X_tg_test[:,0:4],X_scaler_mean,X_scaler_std)).to(device)
        self.y_tg_test = torch.FloatTensor(y_tg_test.reshape(-1,1)).to(device)

        # Density
        self.x_density_train = torch.FloatTensor(self.scaling(X_density_train[:,0:4],X_scaler_mean,X_scaler_std)).to(device)
        self.y_density_train = torch.FloatTensor(y_density_train.reshape(-1,1)).to(device)

        self.x_density_valid = torch.FloatTensor(self.scaling(X_density_valid[:,0:4],X_scaler_mean,X_scaler_std)).to(device)
        self.y_density_valid = torch.FloatTensor(y_density_valid.reshape(-1,1)).to(device)

        self.x_density_test = torch.FloatTensor(self.scaling(X_density_test[:,0:4],X_scaler_mean,X_scaler_std)).to(device)
        self.y_density_test = torch.FloatTensor(y_density_test.reshape(-1,1)).to(device)

        # Optical
        self.x_ri_train = torch.FloatTensor(self.scaling(X_ri_train[:,0:4],X_scaler_mean,X_scaler_std)).to(device)
        self.lbd_ri_train = torch.FloatTensor(lbd_ri_train.reshape(-1,1)).to(device)
        self.y_ri_train = torch.FloatTensor(y_ri_train.reshape(-1,1)).to(device)

        self.x_ri_valid = torch.FloatTensor(self.scaling(X_ri_valid[:,0:4],X_scaler_mean,X_scaler_std)).to(device)
        self.lbd_ri_valid = torch.FloatTensor(lbd_ri_valid.reshape(-1,1)).to(device)
        self.y_ri_valid = torch.FloatTensor(y_ri_valid.reshape(-1,1)).to(device)

        self.x_ri_test = torch.FloatTensor(self.scaling(X_ri_test[:,0:4],X_scaler_mean,X_scaler_std)).to(device)
        self.lbd_ri_test = torch.FloatTensor(lbd_ri_test.reshape(-1,1)).to(device)
        self.y_ri_test = torch.FloatTensor(y_ri_test.reshape(-1,1)).to(device)

        # Raman
        self.x_raman_train = torch.FloatTensor(self.scaling(X_raman_train[:,0:4],X_scaler_mean,X_scaler_std)).to(device)
        self.y_raman_train = torch.FloatTensor(y_raman_train).to(device)

        self.x_raman_valid = torch.FloatTensor(self.scaling(X_raman_valid[:,0:4],X_scaler_mean,X_scaler_std)).to(device)
        self.y_raman_valid = torch.FloatTensor(y_raman_valid).to(device)

    def scaling(self,X,mu,s):
        """perform standard scaling"""
        return(X-mu)/s

    def print_data(self):
        """print the specifications of the datasets"""
        
        print("################################")
        print("#### Dataset specifications ####")
        print("################################")
        
        # print splitting
        size_train = self.x_visco_train.unique(dim=0).shape[0]
        size_valid = self.x_visco_valid.unique(dim=0).shape[0]
        size_test = self.x_visco_test.unique(dim=0).shape[0]
        size_total = size_train+size_valid+size_test

        print("")
        print("Number of data (viscosity): {}".format(size_total))
        print("Number of data in training (viscosity): {}".format(size_train))
        print("Dataset separations are {:.2f} in train, {:.2f} in valid, {:.2f} in test".format(size_train/size_total, 
                                                                                    size_valid/size_total,
                                                                                    size_test/size_total))
        
        size_train = self.x_ri_train.unique(dim=0).shape[0]
        size_valid = self.x_ri_valid.unique(dim=0).shape[0]
        size_test = self.x_ri_test.unique(dim=0).shape[0]
        size_total = size_train+size_valid+size_test

        print("")
        print("Number of data (refractive index): {}".format(size_total))
        print("Number of data in training (refractive index): {}".format(size_train))
        print("Dataset separations are {:.2f} in train, {:.2f} in valid, {:.2f} in test".format(size_train/size_total, 
                                                                                    size_valid/size_total,
                                                                                    size_test/size_total))
        
        size_train = self.x_density_train.unique(dim=0).shape[0]
        size_valid = self.x_density_valid.unique(dim=0).shape[0]
        size_test = self.x_density_test.unique(dim=0).shape[0]
        size_total = size_train+size_valid+size_test

        print("Number of data (density): {}".format(size_total))
        print("Number of data in training (density): {}".format(size_train))
        print("Dataset separations are {:.2f} in train, {:.2f} in valid, {:.2f} in test".format(size_train/size_total, 
                                                                                    size_valid/size_total,
                                                                                    size_test/size_total))
        
        size_train = self.x_raman_train.unique(dim=0).shape[0]
        size_valid = self.x_raman_valid.unique(dim=0).shape[0]
        size_total = size_train+size_valid

        print("Number of data (Raman): {}".format(size_total))
        print("Number of data in training (Raman): {}".format(size_train))
        print("Dataset separations are {:.2f} in train, {:.2f} in valid".format(size_train/size_total, 
                                                                                    size_valid/size_total))

        
        # training shapes
        print("")
        print("This is for checking the consistency of the dataset...")
        
        print("Visco train shape")
        print(self.x_visco_train.shape)
        print(self.T_visco_train.shape)
        print(self.y_visco_train.shape)

        print("Entropy train shape")
        print(self.x_entro_train.shape)
        print(self.y_entro_train.shape)

        print("Tg train shape")
        print(self.x_tg_train.shape)
        print(self.y_tg_train.shape)

        print("Density train shape")
        print(self.x_density_train.shape)
        print(self.y_density_train.shape)

        print("Refactive Index train shape")
        print(self.x_ri_train.shape)
        print(self.lbd_ri_train.shape)
        print(self.y_ri_train.shape)

        print("Raman train shape")
        print(self.x_raman_train.shape)
        print(self.y_raman_train.shape)

        # testing device
        print("")
        print("Where are the datasets? CPU or GPU?")
        
        print("Visco device")
        print(self.x_visco_train.device)
        print(self.T_visco_train.device)
        print(self.y_visco_train.device)

        print("Entropy device")
        print(self.x_entro_train.device)
        print(self.y_entro_train.device)

        print("Tg device")
        print(self.x_tg_train.device)
        print(self.y_tg_train.device)

        print("Density device")
        print(self.x_density_train.device)
        print(self.y_density_train.device)

        print("Refactive Index device")
        print(self.x_ri_test.device)
        print(self.lbd_ri_test.device)
        print(self.y_ri_test.device)

        print("Raman device")
        print(self.x_raman_train.device)
        print(self.y_raman_train.device)

class model(torch.nn.Module):
    """i-MELT model

    """
    def __init__(self, input_size, hidden_size, num_layers, nb_channels_raman,p_drop=0.5, activation_function = torch.nn.ReLU()):
        """Initialization of i-MELT model
        
        Parameters
        ----------
        input_size : int
            number of input parameters
            
        hidden_size : int
            number of hidden units per hidden layer
        
        num_layers : int
            number of hidden layers
        
        nb_channels_raman : int
            number of Raman spectra channels, typically provided by the dataset
        
        p_drop : float (optinal)
            dropout probability, default = 0.5 
        
        activation_function : torch.nn activation function 
            activation function for the hidden units, default = torch.nn.ReLU()
            choose here : https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
        """
        
        
        super(model, self).__init__()

        # init parameters
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.num_layers  = num_layers
        self.nb_channels_raman = nb_channels_raman

        # network related torch stuffs
        self.activation_function = activation_function
        self.dropout = torch.nn.Dropout(p=p_drop)

        self.linears = torch.nn.ModuleList([torch.nn.Linear(input_size, self.hidden_size)])
        self.linears.extend([torch.nn.Linear(self.hidden_size, self.hidden_size) for i in range(1, self.num_layers)])

        self.out_Aterms = torch.nn.Linear(self.hidden_size, 4) # Linear output
        self.out_thermo = torch.nn.ReLU(torch.nn.Linear(self.hidden_size, 13)) # Linear output
        self.out_raman = torch.nn.Linear(self.hidden_size, self.nb_channels_raman) # Linear output

    def output_bias_init(self):
        """bias initialisation for self.out_thermo

        positions are Tg, Sconf(Tg), Ae, A_am, density, fragility (MYEGA one)
        """
        #self.out_thermo.bias = torch.nn.Parameter(data=torch.tensor([np.log(1000.),np.log(10.), # Tg, ScTg
        #                                                             -1.5,-1.5,-1.5, -4.5, # A_AG, A_AM, A_CG, A_TVF
        #                                                             np.log(500.), np.log(100.), np.log(400.), # To_CG, C_CG, C_TVF
         #                                                            np.log(2.3),np.log(25.0), # density, fragility
                                                                     #.90,.20,.98,0.6,0.2,1., # Sellmeier coeffs B1, B2, B3, C1, C2, C3
                                                                     #np.log(1500.) # liquidus
         #                                                            ]))
        
        self.out_Aterms.bias = torch.nn.Parameter(data=torch.tensor([-1.5,-1.5,-1.5, -4.5])) # A_AG, A_AM, A_CG, A_TVF
        self.out_thermo.bias = torch.nn.Parameter(data=torch.tensor([1000.,10., # Tg, ScTg
                                                                     500., 100., 400., # To_CG, C_CG, C_TVF
                                                                     2.3,25.0, # density, fragility
                                                                     .90,.20,.98,0.6,0.2,1., # Sellmeier coeffs B1,B2,B3,C1,C2,C3
                                                                     ]))

    def forward(self, x):
        """foward pass in core neural network"""
        for layer in self.linears: # Feedforward
            x = self.dropout(self.activation_function(layer(x)))
        return x

    def at_gfu(self,x):
        """calculate atom per gram formula unit

        assumes rows are sio2 al2o3 na2o k2o
        """
        out = 3.0*x[:,0] + 5.0*x[:,1] + 3.0*x[:,2] + 3.0*x[:,3]
        return torch.reshape(out, (out.shape[0], 1))

    def aCpl(self,x):
        """calculate term a in equation Cpl = aCpl + bCpl*T
        """
        out = 81.37*x[:,0] + 27.21*x[:,1] + 100.6*x[:,2]  + 50.13*x[:,3] + x[:,0]*(x[:,3]*x[:,3])*151.7
        return torch.reshape(out, (out.shape[0], 1))

    def b_calc(self,x):
        """calculate term b in equation Cpl = aCpl + b*T
        """
        out = 0.09428*x[:,1] + 0.01578*x[:,3]
        return torch.reshape(out, (out.shape[0], 1))

    def ap_calc(self,x):
        """calculate term ap in equation dS = ap ln(T/Tg) + b(T-Tg)
        """
        out = self.aCpl(x) - 3.0*8.314462*self.at_gfu(x)
        return torch.reshape(out, (out.shape[0], 1))

    def dCp(self,x,T):
        out = self.ap_calc(x)*(torch.log(T)-torch.log(self.tg(x))) + self.b_calc(x)*(T-self.tg(x))
        return torch.reshape(out, (out.shape[0], 1))

    def raman_pred(self,x):
        """Raman predicted spectra"""
        return self.out_raman(self.forward(x))

    def ae(self,x):
        """Ae parameter in Adam and Gibbs and MYEGA"""
        out = self.out_Aterms(self.forward(x))[:,0]
        return torch.reshape(out, (out.shape[0], 1))

    def a_am(self,x):
        """A parameter for Avramov-Mitchell"""
        out = self.out_Aterms(self.forward(x))[:,1]
        return torch.reshape(out, (out.shape[0], 1))

    def a_cg(self,x):
        """A parameter for Free Volume (CG)"""
        out = self.out_Aterms(self.forward(x))[:,2]
        return torch.reshape(out, (out.shape[0], 1))

    def a_tvf(self,x):
        """A parameter for Free Volume (CG)"""
        out = self.out_Aterms(self.forward(x))[:,3]
        return torch.reshape(out, (out.shape[0], 1))
    
    def tg(self,x):
        """glass transition temperature Tg"""
        out = self.out_thermo(self.forward(x))[:,0]
        return torch.reshape(out, (out.shape[0], 1))

    def sctg(self,x):
        """configurational entropy at Tg"""
        out = self.out_thermo(self.forward(x))[:,1]
        return torch.reshape(out, (out.shape[0], 1))

    def to_cg(self,x):
        """To parameter for Free Volume (CG)"""
        out = self.out_thermo(self.forward(x))[:,2]
        return torch.reshape(out, (out.shape[0], 1))

    def c_cg(self,x):
        """C parameter for Free Volume (CG)"""
        out = self.out_thermo(self.forward(x))[:,3]
        return torch.reshape(out, (out.shape[0], 1))

    def c_tvf(self,x):
        """C parameter for Free Volume (CG)"""
        out = self.out_thermo(self.forward(x))[:,4]
        return torch.reshape(out, (out.shape[0], 1))

    def density(self,x):
        """glass density"""
        out = self.out_thermo(self.forward(x))[:,5]
        return torch.reshape(out, (out.shape[0], 1))

    def fragility(self,x):
        """melt fragility"""
        out = self.out_thermo(self.forward(x))[:,6]
        return torch.reshape(out, (out.shape[0], 1))

    def S_B1(self,x):
        """Sellmeir B1"""
        out = self.out_thermo(self.forward(x))[:,7]
        return torch.reshape(out, (out.shape[0], 1))

    def S_B2(self,x):
        """Sellmeir B1"""
        out = self.out_thermo(self.forward(x))[:,8]
        return torch.reshape(out, (out.shape[0], 1))

    def S_B3(self,x):
        """Sellmeir B1"""
        out = self.out_thermo(self.forward(x))[:,9]
        return torch.reshape(out, (out.shape[0], 1))

    def S_C1(self,x):
        """Sellmeir C1, with proper scaling"""
        out = 0.01*self.out_thermo(self.forward(x))[:,10]
        return torch.reshape(out, (out.shape[0], 1))

    def S_C2(self,x):
        """Sellmeir C2, with proper scaling"""
        out = 0.1*self.out_thermo(self.forward(x))[:,11]

        return torch.reshape(out, (out.shape[0], 1))

    def S_C3(self,x):
        """Sellmeir C3, with proper scaling"""
        out = 100*self.out_thermo(self.forward(x))[:,12]
        return torch.reshape(out, (out.shape[0], 1))

    def b_cg(self, x):
        """B in free volume (CG) equation"""
        return 0.5*(12.0 - self.a_cg(x)) * (self.tg(x) - self.to_cg(x) + torch.sqrt( (self.tg(x) - self.to_cg(x))**2 + self.c_cg(x)*self.tg(x)))

    def b_tvf(self,x):
        return (12.0-self.a_tvf(x))*(self.tg(x)-self.c_tvf(x))

    def be(self,x):
        """Be term in Adam-Gibbs eq given Ae, Tg and Scong(Tg)"""
        return (12.0-self.ae(x))*(self.tg(x)*self.sctg(x))

    def ag(self,x,T):
        """viscosity from the Adam-Gibbs equation, given chemistry X and temperature T
        """
        return self.ae(x) + self.be(x) / (T* (self.sctg(x) + self.dCp(x, T)))

    def myega(self,x, T):
        """viscosity from the MYEGA equation, given entries X and temperature T
        """
        return self.ae(x) + (12.0 - self.ae(x))*(self.tg(x)/T)*torch.exp((self.fragility(x)/(12.0-self.ae(x))-1.0)*(self.tg(x)/T-1.0))

    def am(self,x, T):
        """viscosity from the Avramov-Mitchell equation, given entries X and temperature T
        """
        return self.a_am(x) + (12.0 - self.a_am(x))*(self.tg(x)/T)**(self.fragility(x)/(12.0 - self.a_am(x)))

    def cg(self,x, T):
        """free volume theory viscosity equation, given entries X and temperature T
        """
        return self.a_cg(x) + 2.0*self.b_cg(x)/(T - self.to_cg(x) + torch.sqrt( (T-self.to_cg(x))**2 + self.c_cg(x)*T))

    def tvf(self,x, T):
        """Tamman-Vogel-Fulscher empirical viscosity, given entries X and temperature T
        """
        return self.a_tvf(x) + self.b_tvf(x)/(T - self.c_tvf(x))

    def sellmeier(self, x, lbd):
        """Sellmeier equation for refractive index calculation, with lbd in microns
        """
        return torch.sqrt( 1.0 + self.S_B1(x)*lbd**2/(lbd**2-self.S_C1(x))
                             + self.S_B2(x)*lbd**2/(lbd**2-self.S_C2(x))
                             + self.S_B3(x)*lbd**2/(lbd**2-self.S_C3(x)))

def training(neuralmodel,ds,criterion,optimizer,save_switch=True,save_name="./temp",train_patience = 50,min_delta=0.1, verbose=True, mode="main"):
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
    save_name : string
        the path to save the model during training

    Options
    -------
    train_patience : int, default = 50
        the number of iterations
    min_delta : float, default = 0.1
        Minimum decrease in the loss to qualify as an improvement,
        a decrease of less than or equal to `min_delta` will count as no improvement.
    verbose : bool, default = True
        Do you want details during training?
    mode : string, default = "main"
        "main" or "pretrain"

    Returns
    -------
    neuralmodel : model
        trained model
    record_train_loss : list
        training loss (global)
    record_valid_loss : list
        validation loss (global)
    """

    if verbose == True:
        time1 = time.time()

        if mode == "pretrain":
            print("! Pretrain mode...\n")
        else:
            print("Full training.\n")

    # scaling coefficients for loss function
    # viscosity is always one
    entro_scale = 1.
    raman_scale = 20.
    density_scale = 1000.
    ri_scale = 10000.
    tg_scale = 0.001

    neuralmodel.train()

    # for early stopping
    epoch = 0
    best_epoch = 0
    val_ex = 0

    # for recording losses
    record_train_loss = []
    record_valid_loss = []

    while val_ex <= train_patience:
        optimizer.zero_grad()

        # Forward pass on training set
        y_ag_pred_train = neuralmodel.ag(ds.x_visco_train,ds.T_visco_train)
        y_myega_pred_train = neuralmodel.myega(ds.x_visco_train,ds.T_visco_train)
        y_am_pred_train = neuralmodel.am(ds.x_visco_train,ds.T_visco_train)
        y_cg_pred_train = neuralmodel.cg(ds.x_visco_train,ds.T_visco_train)
        y_tvf_pred_train = neuralmodel.tvf(ds.x_visco_train,ds.T_visco_train)
        y_raman_pred_train = neuralmodel.raman_pred(ds.x_raman_train)
        y_density_pred_train = neuralmodel.density(ds.x_density_train)
        y_entro_pred_train = neuralmodel.sctg(ds.x_entro_train)
        y_tg_pred_train = neuralmodel.tg(ds.x_tg_train)
        y_cp_pred_train = neuralmodel.dCp(ds.x_entro_train,neuralmodel.tg(ds.x_entro_train))
        y_ri_pred_train = neuralmodel.sellmeier(ds.x_ri_train, ds.lbd_ri_train)
        
        # on validation set
        y_ag_pred_valid = neuralmodel.ag(ds.x_visco_valid,ds.T_visco_valid)
        y_myega_pred_valid = neuralmodel.myega(ds.x_visco_valid,ds.T_visco_valid)
        y_am_pred_valid = neuralmodel.am(ds.x_visco_valid,ds.T_visco_valid)
        y_cg_pred_valid = neuralmodel.cg(ds.x_visco_valid,ds.T_visco_valid)
        y_tvf_pred_valid = neuralmodel.tvf(ds.x_visco_valid,ds.T_visco_valid)
        y_raman_pred_valid = neuralmodel.raman_pred(ds.x_raman_valid)
        y_density_pred_valid = neuralmodel.density(ds.x_density_valid)
        y_entro_pred_valid = neuralmodel.sctg(ds.x_entro_valid)
        y_tg_pred_valid = neuralmodel.tg(ds.x_tg_valid)
        y_cp_pred_valid = neuralmodel.dCp(ds.x_entro_valid,neuralmodel.tg(ds.x_entro_valid))
        y_ri_pred_valid = neuralmodel.sellmeier(ds.x_ri_valid, ds.lbd_ri_valid)
        
        # Compute Loss

        # train
        loss_ag = criterion(y_ag_pred_train, ds.y_visco_train)
        loss_myega = criterion(y_myega_pred_train, ds.y_visco_train)
        loss_am = criterion(y_am_pred_train, ds.y_visco_train)
        loss_cg = criterion(y_cg_pred_train, ds.y_visco_train)
        loss_tvf = criterion(y_tvf_pred_train, ds.y_visco_train)
        loss_raman = criterion(y_raman_pred_train,ds.y_raman_train)
        loss_tg = criterion(y_tg_pred_train,ds.y_tg_train)
        loss_density = criterion(y_density_pred_train,ds.y_density_train)
        loss_entro = criterion(y_entro_pred_train,ds.y_entro_train)
        loss_ri = criterion(y_ri_pred_train,ds.y_ri_train)
        
        if mode == "pretrain":
            loss = tg_scale*loss_tg + raman_scale*loss_raman + density_scale*loss_density + entro_scale*loss_entro + ri_scale*loss_ri 
        else:
            loss = loss_ag + loss_myega + loss_am + loss_cg + loss_tvf + raman_scale*loss_raman + density_scale*loss_density + entro_scale*loss_entro + ri_scale*loss_ri 

        # record global loss
        record_train_loss.append(loss.item())
        
        with torch.set_grad_enabled(False):
            # validation
            loss_ag_v = criterion(y_ag_pred_valid, ds.y_visco_valid)
            loss_myega_v = criterion(y_myega_pred_valid, ds.y_visco_valid)
            loss_am_v = criterion(y_am_pred_valid, ds.y_visco_valid)
            loss_cg_v = criterion(y_cg_pred_valid, ds.y_visco_valid)
            loss_tvf_v = criterion(y_tvf_pred_valid, ds.y_visco_valid)
            loss_raman_v = criterion(y_raman_pred_valid,ds.y_raman_valid)
            loss_tg_v = criterion(y_tg_pred_valid,ds.y_tg_valid)
            loss_density_v = criterion(y_density_pred_valid,ds.y_density_valid)
            loss_entro_v = criterion(y_entro_pred_valid,ds.y_entro_valid)
            loss_ri_v = criterion(y_ri_pred_valid,ds.y_ri_valid)
            
            if mode == "pretrain":
                loss_v = tg_scale*loss_tg_v + raman_scale*loss_raman_v + density_scale*loss_density_v + entro_scale*loss_entro_v + ri_scale*loss_ri_v
            else:
                loss_v = loss_ag_v + loss_myega_v + loss_am_v + loss_cg_v + loss_tvf_v + raman_scale*loss_raman_v + density_scale*loss_density_v + entro_scale*loss_entro_v + ri_scale*loss_ri

        
            record_valid_loss.append(loss_v.item())

        if verbose == True:
            if (epoch % 200 == 0):
              print('Epoch {} => train loss: {}; valid loss: {}'.format(epoch, loss.item(), loss_v.item()))

        # calculating ES criterion
        if epoch == 0:
            val_ex = 0
            best_loss_v = loss_v.item()
        elif loss_v.item() <= best_loss_v - min_delta: # if improvement is significant, this saves the model
            val_ex = 0
            best_epoch = epoch
            best_loss_v = loss_v.item()
            
            # save best model
            if save_switch == True:
                torch.save(neuralmodel.state_dict(), save_name)
        else:
            val_ex += 1

        # Backward pass
        loss.backward()
        optimizer.step()

        epoch += 1

    if verbose == True:
        time2 = time.time()
        print("Running time in seconds:", time2-time1)
        print('Scaled valid loss values are {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f} for Tg, Raman, density, entropy, ri, viscosity (AG)'.format(
        tg_scale*loss_tg_v, raman_scale*loss_raman_v, density_scale*loss_density_v, entro_scale*loss_entro_v,  ri_scale*loss_ri_v, loss_ag_v
        ))

    return neuralmodel, record_train_loss, record_valid_loss

###
### Functions for ternary plots (not really needed with mpltern)
###

def polycorners(ncorners=3):
    '''
    Return 2D cartesian coordinates of a regular convex polygon of a specified
    number of corners.
    Args:
        ncorners (int, optional) number of corners for the polygon (default 3).
    Returns:
        (ncorners, 2) np.ndarray of cartesian coordinates of the polygon.
    '''

    center = np.array([0.5, 0.5])
    points = []

    for i in range(ncorners):
        angle = (float(i) / ncorners) * (np.pi * 2) + (np.pi / 2)
        x = center[0] + np.cos(angle) * 0.5
        y = center[1] + np.sin(angle) * 0.5
        points.append(np.array([x, y]))

    return np.array(points)

def bary2cart(bary, corners):
    '''
    Convert barycentric coordinates to cartesian coordinates given the
    cartesian coordinates of the corners.
    Args:
        bary (np.ndarray): barycentric coordinates to convert. If this matrix
            has multiple rows, each row is interpreted as an individual
            coordinate to convert.
        corners (np.ndarray): cartesian coordinates of the corners.
    Returns:
        2-column np.ndarray of cartesian coordinates for each barycentric
        coordinate provided.
    '''

    cart = None

    if len(bary.shape) > 1 and bary.shape[1] > 1:
        cart = np.array([np.sum(b / np.sum(b) * corners.T, axis=1) for b in bary])
    else:
        cart = np.sum(bary / np.sum(bary) * corners.T, axis=1)

    return cart

def CLR(input_array):
    """Transform chemical composition in colors

    Inputs
    ------
    input_array: n*4 array
        4 chemical inputs with sio2, al2o3, k2o and na2o in 4 columns, n samples in rows

    Returns
    -------
    out: n*3 array
        RGB colors
    """
    XXX = input_array.copy()
    XXX[:,2] = XXX[:,2]+XXX[:,3] # adding alkalis
    out = np.delete(XXX,3,1) # remove 4th row
    # min max scaling to have colors in the full RGB scale
    out[:,0] = (out[:,0]-out[:,0].min())/(out[:,0].max()-out[:,0].min())
    out[:,1] = (out[:,1]-out[:,1].min())/(out[:,1].max()-out[:,1].min())
    out[:,2] = (out[:,2]-out[:,2].min())/(out[:,2].max()-out[:,2].min())
    return out

def chimie_control(data):
    """check that all needed oxides are there and setup correctly the Pandas datalist.
    Parameters
    ----------
    data : Pandas dataframe
        the user input list.
    Returns
    -------
    out : Pandas dataframe
        the output list with all required oxides.
    """
    list_oxides = ["sio2","al2o3","tio2","fe2o3","h2o","li2o","na2o","k2o","mgo","cao","bao","sro","feo","nio","mno","p2o5"]
    datalist = data.copy() # safety network

    for i in list_oxides:
        try:
            oxd = datalist[i]
        except:
            datalist[i] = 0.

    if (datalist["sio2"] > 1).any(): # if values were in percentsifs(self.chimie)

        datalist["sio2"] = datalist["sio2"]/100.0
        datalist["al2o3"] = datalist["al2o3"]/100.0
        datalist["tio2"] = datalist["tio2"]/100.0
        datalist["fe2o3"] = datalist["fe2o3"]/100.0
        datalist["h2o"] = datalist["h2o"]/100.0
        datalist["li2o"] = datalist["li2o"]/100.0
        datalist["na2o"] = datalist["na2o"]/100.0
        datalist["k2o"] = datalist["k2o"]/100.0
        datalist["mgo"] = datalist["mgo"]/100.0
        datalist["cao"] = datalist["cao"]/100.0
        datalist["bao"] = datalist["bao"]/100.0
        datalist["sro"] = datalist["sro"]/100.0
        datalist["feo"] = datalist["feo"]/100.0
        datalist["nio"] = datalist["nio"]/100.0
        datalist["mno"] = datalist["mno"]/100.0
        datalist["p2o5"] = datalist["p2o5"]/100.0

    return datalist
