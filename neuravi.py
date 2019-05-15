########## Calling relevant libraries ##########
import numpy as np

import torch

class model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, layers, nb_channels_raman,p_drop=0.5):
        super(model, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.nb_channels_raman = nb_channels_raman
        self.layers  = layers
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=p_drop)

        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)        
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        
        self.out_thermo = torch.nn.Linear(self.hidden_size, 4) # Linear output
        self.out_raman = torch.nn.Linear(self.hidden_size, self.nb_channels_raman) # Linear output

        # now declaring variables for Ae and Mo
        self.ae = torch.nn.Parameter(data=torch.tensor([-1.5]))
        self.mo = torch.nn.Parameter(data=torch.tensor([21.0]))
    
    def output_bias_init(self):
        """bias initialisation for self.out_thermo"""
        self.out_thermo.bias = torch.nn.Parameter(data=torch.tensor([np.log(1000.),np.log(10.),np.log(30.0),np.log(2.3)]))
    
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
     
                                  
    def forward(self, x):
        """core neural network"""
        hidden1 = self.fc1(x)
        relu1 = self.dropout(self.relu(hidden1))
        hidden2 = self.fc2(relu1)
        relu2 = self.dropout(self.relu(hidden2))
        hidden3 = self.fc3(relu2)
        relu3 = self.dropout(self.relu(hidden3))
        
        return relu3 # return last layer
        
    def raman_pred(self,x):
        """Raman predicted spectra"""
        return self.out_raman(self.forward(x))
      
    def tg(self,x):
        """glass transition temperature"""
        out = torch.exp(self.out_thermo(self.forward(x))[:,0])
        return torch.reshape(out, (out.shape[0], 1))
      
    def sctg(self,x):
        """glass transition temperature"""
        out = torch.exp(self.out_thermo(self.forward(x))[:,1])
        return torch.reshape(out, (out.shape[0], 1))
      
    def fragility(self,x):
        """glass transition temperature"""
        out = torch.exp(self.out_thermo(self.forward(x))[:,2])
        return torch.reshape(out, (out.shape[0], 1))
    
    def density(self,x):
        """glass transition temperature"""
        out = torch.exp(self.out_thermo(self.forward(x))[:,3])
        return torch.reshape(out, (out.shape[0], 1))
    
    def be(self,x):
        return (12.0-self.ae)*(self.tg(x)*self.sctg(x))
      
    def ag(self,x,T):
        """predict viscosity using the Adam-Gibbs equation, given chemistry X and temperature T
        """
        return self.ae + self.be(x) / (T* (self.sctg(x) + self.dCp(x, T)))

    def myega(self,x, T):
        """predict viscosity using the MYEGA equation, given entries X, temperature T, neural network and Ae
        """
        return self.ae + (12.0 - self.ae)*(self.tg(x)/T)*torch.exp((self.fragility(x)/(12.0-self.ae)-1.0)*(self.tg(x)/T-1.0))

    def frag_constraint(self,x):
        """calculated fragility from sctg, ap, b and Mo for constraint addition"""
        out = self.mo*(1.0+(self.ap_calc(x) + self.b_calc(x)*self.tg(x))/self.sctg(x))
        return torch.reshape(out, (out.shape[0], 1))
      
        