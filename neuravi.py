import numpy as np
import torch
import h5py

def data_loader(path_data,path_raman,path_density, device):
    """data loader
    
    Inputs
    ------
    path_data : string
    
    path_raman : string
    
    path_density : string
    
    device : CUDA
    
    Returns
    -------
    Dataset : Dict
    
    """
    
    f = h5py.File(path_data, 'r')
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

    # Dictionary preparation
    dataset = Dict()
    
    # grabbing number of Raman channels
    nb_channels_raman = y_raman_valid.shape[1]

    # preparing data

    # viscosity
    x_visco_train = torch.FloatTensor(X_train[:,0:4]).to(device)
    T_visco_train = torch.FloatTensor(X_train[:,4].reshape(-1,1)).to(device)
    y_visco_train = torch.FloatTensor(y_train[:,0].reshape(-1,1)).to(device)

    x_visco_valid = torch.FloatTensor(X_valid[:,0:4]).to(device)
    T_visco_valid = torch.FloatTensor(X_valid[:,4].reshape(-1,1)).to(device)
    y_visco_valid = torch.FloatTensor(y_valid[:,0].reshape(-1,1)).to(device)

    # entropy
    x_entro_train = torch.FloatTensor(X_entropy_train[:,0:4]).to(device)
    y_entro_train = torch.FloatTensor(y_entropy_train[:,0].reshape(-1,1)).to(device)

    x_entro_valid = torch.FloatTensor(X_entropy_valid[:,0:4]).to(device)
    y_entro_valid = torch.FloatTensor(y_entropy_valid[:,0].reshape(-1,1)).to(device)

    # tg
    x_tg_train = torch.FloatTensor(X_tg_train[:,0:4]).to(device)
    y_tg_train = torch.FloatTensor(y_tg_train.reshape(-1,1)).to(device)

    x_tg_valid = torch.FloatTensor(X_tg_valid[:,0:4]).to(device)
    y_tg_valid = torch.FloatTensor(y_tg_valid.reshape(-1,1)).to(device)

    # Density
    x_density_train = torch.FloatTensor(X_density_train[:,0:4]).to(device)
    y_density_train = torch.FloatTensor(y_density_train.reshape(-1,1)).to(device)

    x_density_valid = torch.FloatTensor(X_density_valid[:,0:4]).to(device)
    y_density_valid = torch.FloatTensor(y_density_valid.reshape(-1,1)).to(device)

    # Raman
    x_raman_train = torch.FloatTensor(X_raman_train[:,0:4]).to(device)
    y_raman_train = torch.FloatTensor(y_raman_train).to(device)

    x_raman_valid = torch.FloatTensor(X_raman_valid[:,0:4]).to(device)
    y_raman_valid = torch.FloatTensor(y_raman_valid).to(device)

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
            
#         self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)        
#         self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
#         self.fc3 = torch.nn.Linear(self.hidden_size, self.hidden_size)

        self.out_thermo = torch.nn.Linear(self.hidden_size, 6) # Linear output
        self.out_raman = torch.nn.Linear(self.hidden_size, self.nb_channels_raman) # Linear output
    
    def forward(self, x):
        """core neural network"""
        # Feedforward
        for layer in self.linears:
            x = self.dropout(self.relu(layer(x)))
        return x
#         hidden1 = self.fc1(x)
#         relu1 = self.dropout(self.relu(hidden1))
#         hidden2 = self.fc2(relu1)
#         relu2 = self.dropout(self.relu(hidden2))
#         hidden3 = self.fc3(relu2)
#         relu3 = self.dropout(self.relu(hidden3))        
#         return relu3 # return last layer
    
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
        """viscosity frself.linears = nn.ModuleList([nn.Linear(input_size, layers_size)])
     self.linears.extend([nn.Linear(layers_size, layers_size) for i in range(1, self.num_layers-1)])
    om the MYEGA equation, given entries X and temperature T
        """
        return self.ae(x) + (12.0 - self.ae(x))*(self.tg(x)/T)*torch.exp((self.fragility(x)/(12.0-self.ae(x))-1.0)*(self.tg(x)/T-1.0))
      
    def am(self,x, T):
        """viscosity from the Avramov-Mitchell equation, given entries X and temperature T
        """
        return self.a_am(x) + (12.0 - self.a_am(x))*(self.tg(x)/T)**(self.fragility(x)/12.0)
    