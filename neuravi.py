import numpy as np
import torch, time
import h5py

class data_loader():
    """custom data loader for batch training

    """
    def __init__(self,path_viscosity,path_raman,path_density, device):
        """Inputs
        ------
        path_viscosity : string

        path_raman : string

        path_density : string

        device : CUDA"""
        f = h5py.File(path_viscosity, 'r')

        # List all groups
        self.X_columns = f['X_columns'].value

        # Entropy dataset
        X_entropy_train = f["X_entropy_train"].value
        y_entropy_train = f["y_entropy_train"].value

        X_entropy_valid = f["X_entropy_valid"].value
        y_entropy_valid = f["y_entropy_valid"].value

        X_entropy_test = f["X_entropy_test"].value
        y_entropy_test = f["y_entropy_test"].value

        # Viscosity dataset
        X_train = f["X_train"].value
        y_train = f["y_train"].value

        X_valid = f["X_valid"].value
        y_valid = f["y_valid"].value

        X_test = f["X_test"].value
        y_test = f["y_test"].value

        # Tg dataset
        X_tg_train = f["X_tg_train"].value
        X_tg_valid= f["X_tg_valid"].value
        X_tg_test = f["X_tg_test"].value

        y_tg_train = f["y_tg_train"].value
        y_tg_valid = f["y_tg_valid"].value
        y_tg_test = f["y_tg_test"].value

        f.close()

        # Raman dataset
        f = h5py.File(path_raman, 'r')
        X_raman_train = f["X_raman_train"].value
        y_raman_train = f["y_raman_train"].value
        X_raman_valid = f["X_raman_test"].value
        y_raman_valid = f["y_raman_test"].value
        f.close()

        # Density dataset
        f = h5py.File(path_density, 'r')
        X_density_train = f["X_density_train"].value
        X_density_valid = f["X_density_valid"].value
        X_density_test = f["X_density_test"].value

        y_density_train = f["y_density_train"].value
        y_density_valid = f["y_density_valid"].value
        y_density_test = f["y_density_test"].value
        f.close()

        # grabbing number of Raman channels
        self.nb_channels_raman = y_raman_valid.shape[1]

        # preparing data for pytorch

        # Scaler
        X_scaler_mean = 0.0#np.mean(X_train[:,0:4], axis=0)
        X_scaler_std = 1.0#np.std(X_train[:,0:4], axis=0)

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

        # Raman
        self.x_raman_train = torch.FloatTensor(self.scaling(X_raman_train[:,0:4],X_scaler_mean,X_scaler_std)).to(device)
        self.y_raman_train = torch.FloatTensor(y_raman_train).to(device)

        self.x_raman_valid = torch.FloatTensor(self.scaling(X_raman_valid[:,0:4],X_scaler_mean,X_scaler_std)).to(device)
        self.y_raman_valid = torch.FloatTensor(y_raman_valid).to(device)

    def scaling(self,X,mu,s):
        return(X-mu)/s

    def print_data(self):
        # testing shapes
        print("Visco shape")
        print(self.x_visco_train.shape)
        print(self.T_visco_train.shape)
        print(self.y_visco_train.shape)

        print("Entropy shape")
        print(self.x_entro_train.shape)
        print(self.y_entro_train.shape)

        print("Tg shape")
        print(self.x_tg_train.shape)
        print(self.y_tg_train.shape)

        print("Density shape")
        print(self.x_density_train.shape)
        print(self.y_density_train.shape)

        print("Raman shape")
        print(self.x_raman_train.shape)
        print(self.y_raman_train.shape)

        # testing device
        print("Visco device")
        print(self.x_visco_train.device)
        print(self.T_visco_train.device)
        print(self.y_visco_train.device)

        print("Entropy device")
        print(self.x_entro_train.device)
        print(self.y_entro_train.device)

        print("Tg shape")
        print(self.x_tg_train.device)
        print(self.y_tg_train.device)

        print("Density device")
        print(self.x_density_train.device)
        print(self.y_density_train.device)

        print("Raman device")
        print(self.x_raman_train.device)
        print(self.y_raman_train.device)

class model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, nb_channels_raman,p_drop=0.5):
        super(model, self).__init__()

        # init parameters
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.num_layers  = num_layers
        self.nb_channels_raman = nb_channels_raman

        # network related torch stuffs
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=p_drop)

        self.linears = torch.nn.ModuleList([torch.nn.Linear(input_size, self.hidden_size)])
        self.linears.extend([torch.nn.Linear(self.hidden_size, self.hidden_size) for i in range(1, self.num_layers)])

        self.out_thermo = torch.nn.Linear(self.hidden_size, 6) # Linear output
        self.out_raman = torch.nn.Linear(self.hidden_size, self.nb_channels_raman) # Linear output
        
        self.AAA = torch.nn.Parameter(data=torch.tensor([25.0,15.0]))

    def forward(self, x):
        """core neural network"""
        # Feedforward
        for layer in self.linears:
            x = self.dropout(self.relu(layer(x)))
        return x

    def output_bias_init(self):
        """bias initialisation for self.out_thermo

        positions are Tg, Sconf(Tg), Ae, A_am, density, fragility (MYEGA one)
        """
        self.out_thermo.bias = torch.nn.Parameter(data=torch.tensor([np.log(1000.),np.log(10.),-1.5,-1.5,np.log(2.3),np.log(20.0)]))

    def at_gfu(self,x):
        """calculate atom per gram formula unit

        assumes rows are sio2 al2o3 na2o k2o
        """
        out = 3.0*x[:,0] + 5.0*x[:,1] + 3.0*x[:,2] + 3.0*x[:,3]
        return torch.reshape(out, (out.shape[0], 1))

    def aCpl(self,x):
        """calculate term a in equation Cpl = qCpl + bCpl*T
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

    def tg(self,x):
        """glass transition temperature Tg"""
        out = torch.exp(self.out_thermo(self.forward(x))[:,0])
        return torch.reshape(out, (out.shape[0], 1))

    def sctg(self,x):
        """configurational entropy at Tg"""
        out = torch.exp(self.out_thermo(self.forward(x))[:,1])
        return torch.reshape(out, (out.shape[0], 1))

    def ae(self,x):
        """Ae parameter in Adam and Gibbs and MYEGA"""
        out = self.out_thermo(self.forward(x))[:,2]
        return torch.reshape(out, (out.shape[0], 1))

    def a_am(self,x):
        """A parameter for Avramov-Mitchell"""
        out = self.out_thermo(self.forward(x))[:,3]
        return torch.reshape(out, (out.shape[0], 1))

    def density(self,x):
        """glass density"""
        out = torch.exp(self.out_thermo(self.forward(x))[:,4])
        return torch.reshape(out, (out.shape[0], 1))

    def fragility(self,x):
        """melt fragility"""
        out = torch.exp(self.out_thermo(self.forward(x))[:,5])
        return torch.reshape(out, (out.shape[0], 1))

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

def pretraining(neuralmodel,ds,criterion,optimizer, verbose=True):
    if verbose == True:
        print("Pretrain...\n")

        time1 = time.time()

    neuralmodel.train()

    # for early stopping
    pretrain_patience = 50
    epoch = 0
    best_epoch = 0
    val_ex = 0

    # for recording losses
    record_pretrain_loss = []
    record_prevalid_loss = []

    while val_ex <= pretrain_patience:

        optimizer.zero_grad()

        # Forward pass
        y_raman_pred_train = neuralmodel.raman_pred(ds.x_raman_train)
        y_density_pred_train = neuralmodel.density(ds.x_density_train)
        y_tg_pred_train = neuralmodel.tg(ds.x_tg_train)
        y_entro_pred_train = neuralmodel.sctg(ds.x_entro_train)

        # on validation set
        y_raman_pred_valid = neuralmodel.raman_pred(ds.x_raman_valid)
        y_density_pred_valid = neuralmodel.density(ds.x_density_valid)
        y_tg_pred_valid = neuralmodel.tg(ds.x_tg_valid)
        y_entro_pred_valid = neuralmodel.sctg(ds.x_entro_valid)

        # Compute Loss

        # train
        loss_tg = criterion(y_tg_pred_train, ds.y_tg_train)
        loss_raman = criterion(y_raman_pred_train,ds.y_raman_train)
        loss_density = criterion(y_density_pred_train,ds.y_density_train)
        loss_entro = criterion(y_entro_pred_train,ds.y_entro_train)

        loss = 0.001*loss_tg + 10*loss_raman + 1000*loss_density + loss_entro

        # validation
        loss_tg_v = criterion(y_tg_pred_valid, ds.y_tg_valid)
        loss_raman_v = criterion(y_raman_pred_valid,ds.y_raman_valid)
        loss_density_v = criterion(y_density_pred_valid,ds.y_density_valid)
        loss_entro_v = criterion(y_entro_pred_valid,ds.y_entro_valid)

        loss_v = 0.001*loss_tg_v + 10*loss_raman_v + 1000*loss_density_v + loss_entro_v

        record_pretrain_loss.append(loss.item())
        record_prevalid_loss.append(loss_v.item())

        if verbose == True:
            if (epoch % 100 == 0):
              print('Epoch {} => train loss: {}; valid loss: {}'.format(epoch, loss.item(), loss_v.item()))

        # calculating early-stopping criterion
        if epoch == 0:
            val_ex = 0
            best_loss_v = loss_v.item()
        elif loss_v.item() <= best_loss_v:
            val_ex = 0
            best_epoch = epoch
            best_loss_v = loss_v.item()
        else:
            val_ex += 1

        # Backward pass
        loss.backward()
        optimizer.step()

        epoch += 1

    if verbose == True:
        time2 = time.time()

        print("Running time in seconds:", time2-time1)

    return neuralmodel, record_pretrain_loss, record_prevalid_loss

def maintraining(neuralmodel,ds,criterion,optimizer,save_name,train_patience = 50,verbose=True):
    if verbose == True:
        time1 = time.time()

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

        # Forward pass
        y_ag_pred_train = neuralmodel.ag(ds.x_visco_train,ds.T_visco_train)
        y_myega_pred_train = neuralmodel.myega(ds.x_visco_train,ds.T_visco_train)
        y_am_pred_train = neuralmodel.am(ds.x_visco_train,ds.T_visco_train)
        y_raman_pred_train = neuralmodel.raman_pred(ds.x_raman_train)
        y_density_pred_train = neuralmodel.density(ds.x_density_train)
        y_entro_pred_train = neuralmodel.sctg(ds.x_entro_train)
        y_cp_pred_train = neuralmodel.dCp(ds.x_entro_train,neuralmodel.tg(ds.x_entro_train))
        
        # on validation set
        y_ag_pred_valid = neuralmodel.ag(ds.x_visco_valid,ds.T_visco_valid)
        y_myega_pred_valid = neuralmodel.myega(ds.x_visco_valid,ds.T_visco_valid)
        y_am_pred_valid = neuralmodel.am(ds.x_visco_valid,ds.T_visco_valid)
        y_raman_pred_valid = neuralmodel.raman_pred(ds.x_raman_valid)
        y_density_pred_valid = neuralmodel.density(ds.x_density_valid)
        y_entro_pred_valid = neuralmodel.sctg(ds.x_entro_valid)
        y_cp_pred_valid = neuralmodel.dCp(ds.x_entro_valid,neuralmodel.tg(ds.x_entro_valid))
        
        # Compute Loss

        # train
        loss_ag = criterion(y_ag_pred_train, ds.y_visco_train)
        loss_myega = criterion(y_myega_pred_train, ds.y_visco_train)
        loss_am = criterion(y_am_pred_train, ds.y_visco_train)
        loss_raman = criterion(y_raman_pred_train,ds.y_raman_train)
        loss_density = criterion(y_density_pred_train,ds.y_density_train)
        loss_entro = criterion(y_entro_pred_train,ds.y_entro_train)
        loss_ = criterion(y_entro_pred_train,y_cp_pred_train/((neuralmodel.fragility(ds.x_entro_train)-neuralmodel.AAA[1])/neuralmodel.AAA[0]))
        
        loss = loss_ag + loss_myega + loss_am + 10*loss_raman + 1000*loss_density + loss_entro #+ loss_#

        # validation
        loss_ag_v = criterion(y_ag_pred_valid, ds.y_visco_valid)
        loss_myega_v = criterion(y_myega_pred_valid, ds.y_visco_valid)
        loss_am_v = criterion(y_am_pred_valid, ds.y_visco_valid)
        loss_raman_v = criterion(y_raman_pred_valid,ds.y_raman_valid)
        loss_density_v = criterion(y_density_pred_valid,ds.y_density_valid)
        loss_entro_v = criterion(y_entro_pred_valid,ds.y_entro_valid)

        loss_v = loss_ag_v + loss_myega_v + loss_am_v + 10*loss_raman_v + 1000*loss_density_v + loss_entro_v

        record_train_loss.append(loss.item())
        record_valid_loss.append(loss_v.item())

        if verbose == True:
            if (epoch % 200 == 0):
              print('Epoch {} => train loss: {}; valid loss: {}'.format(epoch, loss.item(), loss_v.item()))

        # calculating ES criterion
        if epoch == 0:
            val_ex = 0
            best_loss_v = loss_v.item()
        elif loss_v.item() <= best_loss_v:
            val_ex = 0
            best_epoch = epoch
            best_loss_v = loss_v.item()

            # save best model
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

    return neuralmodel, record_train_loss, record_valid_loss
