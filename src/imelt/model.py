# (c) Charles Le Losq et co 2022-2024
# see embedded licence file
# imelt V2.1

import numpy as np
import torch, time
import pandas as pd
from scipy.constants import Avogadro, Planck
import imelt as imelt
import warnings

# to load data and models in a library
from pathlib import Path
import os

_BASEMODELPATH = Path(os.path.dirname(__file__)) / "models"

###
### MODEL
###
class model(torch.nn.Module):
    """i-MELT model"""

    def __init__(
        self,
        input_size,
        hidden_size=300,
        num_layers=4,
        nb_channels_raman=800,
        p_drop=0.2,
        activation_function=torch.nn.GELU(),
    ):
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
            dropout probability, default = 0.2

        activation_function : torch.nn activation function (optional)
            activation function for the hidden units, default = torch.nn.GELU()

        """
        super(model, self).__init__()

        #
        # init parameters
        #
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nb_channels_raman = nb_channels_raman
        self.activation_function = activation_function
        self.p_drop = p_drop
        self.dropout = torch.nn.Dropout(p=p_drop)

        #
        # general shape of the network
        #
        self.linears = torch.nn.ModuleList(
            [torch.nn.Linear(input_size, self.hidden_size)]
        )
        self.linears.extend(
            [
                torch.nn.Linear(self.hidden_size, self.hidden_size)
                for i in range(1, self.num_layers)
            ]
        )

        ###
        # output layers
        ###
        self.out_thermo = torch.nn.Linear(
            self.hidden_size, 34
        )  # Linear output, 22 without Cp
        self.out_raman = torch.nn.Linear(
            self.hidden_size, self.nb_channels_raman
        )  # Linear output

        # the model will also contains parameter for the losses
        # this was adjusted to the dataset used in the paper
        self.log_vars = torch.nn.Parameter(
            torch.tensor(
                [
                    np.log(0.25**2),  # viscosity
                    np.log(0.05**2),  # raman
                    np.log(0.02**2),  # density
                    np.log(0.50**2),  # entropy
                    np.log(0.004**2),  # refractive index
                    np.log(4.0**2),  # Cpl
                    np.log(6**2),  # Elastic modulus #10
                    np.log(2**2),  # CTE #5
                    np.log(3**2),  # Abbe Number #5
                    np.log(40.0**2),  # liquidus # 50
                ]
            ),
            requires_grad=True,
        )

    def output_bias_init(self):
        """bias initialisation for self.out_thermo

        """
        self.out_thermo.bias = torch.nn.Parameter(
            data=torch.tensor(
                [
                    np.log(1000.0), # Tg
                    np.log(8.0),  # ScTg
                    -3.5, # A_AG
                    -3.5, # A_AM
                    -3.5, # A_CG
                    -4.5, # A_TVF
                    np.log(715.0), # To_CG
                    np.log(61.0), # C_CG
                    np.log(500.0), # C_TVF
                    np.log(27.29), # SiO2 molar volume
                    np.log(36.66), # al2o3 molar volume
                    np.log(29.65), # na2o molar volume
                    np.log(47.28), # k2o molar volume
                    np.log(12.66), # mgo molar volume
                    np.log(20.66), # cao molar volume
                    np.log(44.0),  # fragility
                    0.90, # Sellmeier coeff B1
                    0.20, # Sellmeier coeff B2
                    0.98, # Sellmeier coeff B3
                    0.6, # Sellmeier coeff C1
                    0.2, # Sellmeier coeff C2
                    1.0, # Sellmeier coeff C3
                    np.log(81.37), # SiO2 Cpi
                    np.log(130.2), # al2o3 Cpi
                    np.log(100.6), # na2o Cpi
                    np.log(50.13), # k2o Cpi
                    np.log(85.78), # mgo Cpi
                    np.log(86.05), # cao Cpi
                    np.log(0.03), # Al2O3 excess T Cpi
                    np.log(0.01578), # K2O excess T Cpi
                    np.log(100.0),  # glass elastic modulus
                    np.log(8.5),  # CTE
                    np.log(57.0),  # Abbe Number
                    np.log(1585.0),  # liquidus temperature
                ]
            )
        )

    def use_pretrained_model(self, pretrained_name, same_input_size=False):
        """to use a pretrained model"""
        if same_input_size == False:
            pretrained_model = torch.load(pretrained_name)
            pretrained_model["old_linear.0.weight"] = pretrained_model.pop(
                "linears.0.weight"
            )
            self.load_state_dict(pretrained_model, strict=False)
        else:
            pretrained_model = torch.load(pretrained_name)
            self.load_state_dict(pretrained_model, strict=False)

    def forward(self, x):
        """foward pass in core neural network"""
        for layer in self.linears:  # Feedforward
            x = self.dropout(self.activation_function(layer(x)))
        return x
        
    def _quick_reshape(self, _input, exp=False):
        """reshape of vectors for getting nx1 tensor
        
        if exp == True, returns the exponential"""
        out = torch.reshape(_input, (_input.shape[0], 1))
        if exp == True:
            return torch.exp(out)
        else:
            return out
    
    def predict_all(self, x, T=None, lbd=None):
        """Compute all properties in one forward pass"""
        hidden = self.forward(x)
        thermo_out = self.out_thermo(hidden)
        
        # Compute all scalar properties
        properties = {
            'tg': self._quick_reshape(thermo_out[:, 0], exp=True),
            'sctg': self._quick_reshape(thermo_out[:, 1], exp=True),
            'ae': self._quick_reshape(thermo_out[:, 2]),
            'a_am': self._quick_reshape(thermo_out[:,3]),
            'a_cg': self._quick_reshape(thermo_out[:,4]),
            'a_tvf': self._quick_reshape(thermo_out[:,5]),
            'to_cg' : self._quick_reshape(thermo_out[:,6], exp=True), # To parameter for Free Volume (CG)
            'c_cg' : self._quick_reshape(thermo_out[:,7], exp=True), # C parameter for Free Volume (CG)
            'c_tvf' : self._quick_reshape(thermo_out[:,8], exp=True), # C parameter for VFT
            'vm_glass': torch.exp(thermo_out[:, 9:15]), #partial molar volume of oxide cations in glass
            'fragility': self._quick_reshape(thermo_out[:, 15], exp=True),
            'S_B1': self._quick_reshape(thermo_out[:,16]),
            'S_B2': self._quick_reshape(thermo_out[:,17]),
            'S_B3': self._quick_reshape(thermo_out[:,18]),
            'S_C1': 0.01*self._quick_reshape(thermo_out[:,19]), # scaled
            'S_C2': 0.1*self._quick_reshape(thermo_out[:,20]), # scaled
            'S_C3': 100.0*self._quick_reshape(thermo_out[:,21]), # scaled
            # 6 values in order: SiO2 Al2O3 Na2O K2O MgO CaO
            # 2 last values are for bCPl for Al2O3 and K2O
            'partial_cpl': torch.exp(thermo_out[:, 22:30]),
            'a_cp' : torch.exp(thermo_out[:, 22:28]),
            'b_cp' : torch.exp(thermo_out[:, 28:30]),
            'elastic_modulus': self._quick_reshape(thermo_out[:,30], exp=True),
            'cte': self._quick_reshape(thermo_out[:,31], exp=True),
            'abbe': self._quick_reshape(thermo_out[:,32], exp=True),
            'liquidus': self._quick_reshape(thermo_out[:,33], exp=True),
            'raman_spectra': self.out_raman(hidden)
        }

        # calculate density
        properties['density_glass'] = self._calculate_density_glass(x, 
                                                                    properties["vm_glass"])

        # term ap in equation dS = ap ln(T/Tg) + b(T-Tg)
        out = self.aCpl(x, properties["a_cp"]) - self.cpg_tg(x)
        properties['ap_calc'] = torch.reshape(out, (out.shape[0], 1))

        # a_Cpl and b_Cpl
        properties['aCpl'] = self.aCpl(x, properties["a_cp"])
        properties['bCpl'] = self.bCpl(x, properties["b_cp"])

        # calculate b terms for the various viscosity eqs.
        properties['be'] = (12.0 - properties['ae']) * (properties['tg'] * properties['sctg'])
        properties['b_cg'] = (
                0.5
                * (12.0 - properties['a_cg'])
                * (
                    properties['tg'] 
                    - properties['to_cg'] 
                    + torch.sqrt(
                        (properties['tg'] - properties['to_cg']) ** 2 + properties['c_cg'] * properties['tg']
                        )
                  )
            )
        properties['b_tvf'] = (12.0 - properties['a_tvf']) * (properties['tg'] - properties['c_tvf'])
        
        # Compute dependent properties
        if T is not None:
            # Liquid heat capacity at T
            out = properties['aCpl'] + properties['bCpl'] * T
            properties['cpl'] = torch.reshape(out, (out.shape[0], 1))

            # delta Cp = Cp conf
            out = properties['ap_calc'] * (torch.log(T) - torch.log(properties['tg'])) + self.bCpl(x, properties["b_cp"]) * (T - properties['tg'])
            properties['dCp'] = torch.reshape(out, (out.shape[0], 1))

            # calculate viscosity
            properties['ag'] = self._calculate_ag(T, 
                                                  properties['ae'],
                                                  properties['be'],
                                                  properties['sctg'],
                                                  properties['dCp'])

            properties['myega'] = self._calculate_myega(T,
                                                        properties['tg'],
                                                        properties['ae'],
                                                        properties['fragility'])
            
            properties['am'] = self._calculate_am(T,
                                                  properties['tg'],
                                                  properties['a_am'],
                                                  properties['fragility'])
            
            properties['cg'] = self._calculate_cg(T,
                                                  properties['a_cg'],
                                                  properties['b_cg'],
                                                  properties['to_cg'],
                                                  properties['c_cg'])
            
            properties['tvf'] = self._calculate_tvf(T,
                                                    properties['tg'],
                                                    properties['a_tvf'],
                                                    properties['c_tvf'])
            
        
        if lbd is not None:
            properties['sellmeier'] = self._calculate_sellmeier(lbd,
                                                                properties['S_B1'],
                                                                properties['S_B2'],
                                                                properties['S_B3'],
                                                                properties['S_C1'],
                                                                properties['S_C2'],
                                                                properties['S_C3'])
        
        return properties
    
    def _calculate_ag(self, T, ae, be, sctg, dCp):
        """Adam-Gibbs equation for viscosity calculation"""
        return ae + be / (T * (sctg + dCp))

    def _calculate_myega(self, T, tg, ae, fragility):
        """viscosity from MYEGA equation"""
        return ae + (12.0 - ae) * (tg / T) * torch.exp(
            (fragility / (12.0 - ae) - 1.0) * (tg / T - 1.0)
        )
    
    def _calculate_am(self, T, tg, a_am, fragility):
        """Avramov-Mitchell equation for viscosity calculation"""
        return a_am + (12.0 - a_am) * (tg / T) ** (fragility / (12.0 - a_am))
    
    def _calculate_cg(self, T, a_cg, b_cg, to_cg, c_cg):
        """Cohen Grest (free volume) viscosity equation"""
        return a_cg + 2.0 * b_cg / (T - to_cg + torch.sqrt((T - to_cg) ** 2 + c_cg * T))
    
    def _calculate_tvf(self, T, tg, a_tvf, c_tvf):
        """Tamman-Vogel-Fulscher empirical viscosity"""
        return a_tvf + ((12.0 - a_tvf) * (tg - c_tvf)) / (T - c_tvf)

    def _calculate_sellmeier(self, lbd, S_B1, S_B2, S_B3, S_C1, S_C2, S_C3):
        """Sellmeier equation for refractive index calculation, with lbd in microns"""
        return torch.sqrt(
            1.0
            + S_B1 * lbd**2 / (lbd**2 - S_C1)
            + S_B2 * lbd**2 / (lbd**2 - S_C2)
            + S_B3 * lbd**2 / (lbd**2 - S_C3)
        )

    def at_gfu(self, x):
        """calculate atom per gram formula unit

        assumes first columns are sio2 al2o3 na2o k2o mgo cao
        """
        out = (
            3.0 * x[:, 0]
            + 5.0 * x[:, 1]
            + 3.0 * x[:, 2]
            + 3.0 * x[:, 3]
            + 2.0 * x[:, 4]
            + 2.0 * x[:, 5]
        )
        return torch.reshape(out, (out.shape[0], 1))

    def aCpl(self, x, a_cp):
        """calculate term a in equation Cpl = aCpl + bCpl*T

        Partial molar Cp are from the neural network.

        assumes first columns are sio2 al2o3 na2o k2o mgo cao
        """
        out = (
            a_cp[:, 0] * x[:, 0]  # Cp liquid SiO2, fixed value from Richet 1984
            + a_cp[:, 1] * x[:, 1]  # Cp liquid Al2O3
            + a_cp[:, 2] * x[:, 2]  # Cp liquid Na2O
            + a_cp[:, 3] * x[:, 3]  # Cp liquid K2O
            + a_cp[:, 4] * x[:, 4]  # Cp liquid MgO
            + a_cp[:, 5] * x[:, 5]  # Cp liquid CaO)
        )

        return torch.reshape(out, (out.shape[0], 1))

    def bCpl(self, x, b_cp):
        """calculate term b in equation Cpl = aCpl + bCpl*T

        assumes first columns are sio2 al2o3 na2o k2o mgo cao

        only apply B terms on Al and K
        """
        # euqation from Richet 1985, dependency on Al2O3 and K2O
        out = b_cp[:, 0] * x[:, 1] + b_cp[:, 1] * x[:, 3]

        return torch.reshape(out, (out.shape[0], 1))

    def cpg_tg(self, x):
        """Glass heat capacity at Tg calculated from Dulong and Petit limit"""
        return 3.0 * 8.314462 * self.at_gfu(x)

    def cpl(self, x, T):
        """Liquid heat capacity at T"""
        return self.predict_all(x, T=T)['cpl']

    def partial_cpl(self, x):
        """partial molar values for Cpl
        6 values in order: SiO2 Al2O3 Na2O K2O MgO CaO
        2 last values are temperature dependence for Al2O3 and K2O
        """
        return self.predict_all(x)['partial_cpl']

    def ap_calc(self, x):
        """calculate term ap in equation dS = ap ln(T/Tg) + b(T-Tg)"""
        return self.predict_all(x)['ap_calc']

    def dCp(self, x, T):
        """calculate Cpconf""" 
        return self.predict_all(x, T=T)['dCp']

    def raman_pred(self, x):
        """Raman predicted spectra"""
        return self.out_raman(self.forward(x))

    def tg(self, x):
        """glass transition temperature Tg"""
        return self.predict_all(x)['tg']

    def sctg(self, x):
        """configurational entropy at Tg"""
        return self.predict_all(x)['sctg']

    def ae(self, x):
        """Ae parameter in Adam and Gibbs and MYEGA"""
        return self.predict_all(x)['ae']

    def a_am(self, x):
        """A parameter for Avramov-Mitchell"""
        return self.predict_all(x)['a_am']

    def a_cg(self, x):
        """A parameter for Free Volume (CG)"""
        return self.predict_all(x)['a_cg']

    def a_tvf(self, x):
        """A parameter for VFT"""
        return self.predict_all(x)['a_tvf']

    def to_cg(self, x):
        """To parameter for Free Volume (CG)"""
        return self.predict_all(x)['to_cg']

    def c_cg(self, x):
        """C parameter for Free Volume (CG)"""
        return self.predict_all(x)['c_cg']

    def c_tvf(self, x):
        """C parameter for VFT"""
        return self.predict_all(x)['c_tvf']

    def vm_glass(self, x):
        """partial molar volume of oxide cations in glass"""
        return self.predict_all(x)['vm_glass']

    def _calculate_density_glass(self, x, vm_):
        """glass density

        X columns are sio2 al2o3 na2o k2o mgo cao

        vm_ are the partial molar volumes of the oxides, same shape as X
        """
        w = imelt.molarweights()  # weights

        # calculation of glass molar volume
        v_g = (
            vm_[:, 0] * x[:, 0] # sio2
            + vm_[:, 1] * x[:, 1]  # al2o3
            + vm_[:, 2] * x[:, 2]  # na2o
            + vm_[:, 3] * x[:, 3]  # k2o
            + vm_[:, 4] * x[:, 4]  # mgo
            + vm_[:, 5] * x[:, 5] # cao
        )

        # glass mass for one mole of oxides
        XMW_SiO2 = x[:, 0] * w["sio2"]
        XMW_Al2O3 = x[:, 1] * w["al2o3"]
        XMW_Na2O = x[:, 2] * w["na2o"]
        XMW_K2O = x[:, 3] * w["k2o"]
        XMW_MgO = x[:, 4] * w["mgo"]
        XMW_CaO = x[:, 5] * w["cao"]

        XMW_tot = XMW_SiO2 + XMW_Al2O3 + XMW_Na2O + XMW_K2O + XMW_MgO + XMW_CaO
        XMW_tot = XMW_tot.reshape(-1, 1)

        out = XMW_tot / v_g.reshape(-1, 1)
        return torch.reshape(out, (out.shape[0], 1))

    def density_glass(self, x):
        """glass density

        X columns are sio2 al2o3 na2o k2o mgo cao
        """
        return self.predict_all(x)['density_glass']

    def density_melt(self, x, T, P=1):
        """melt density, calculated as in DensityX"""

        # grab constants
        d_cts = constants()
        w = imelt.molarweights()

        # mass for one mole of oxides
        XMW_SiO2 = x[:, 0] * w["sio2"]
        XMW_Al2O3 = x[:, 1] * w["al2o3"]
        XMW_Na2O = x[:, 2] * w["na2o"]
        XMW_K2O = x[:, 3] * w["k2o"]
        XMW_MgO = x[:, 4] * w["mgo"]
        XMW_CaO = x[:, 5] * w["cao"]

        XMW_tot = XMW_SiO2 + XMW_Al2O3 + XMW_Na2O + XMW_K2O + XMW_MgO + XMW_CaO
        XMW_tot = XMW_tot.reshape(-1, 1)

        # calculation of corrected VM
        c_Vm_Tref = (
            x[:, 0] * d_cts.c_sio2
            + x[:, 1] * d_cts.c_al2o3
            + x[:, 2] * d_cts.c_na2o
            + x[:, 3] * d_cts.c_k2o
            + x[:, 4] * d_cts.c_mgo
            + x[:, 5] * d_cts.c_cao
        )

        # calculation of alphas
        alpha_ = (
            x[:, 0] * d_cts.dVdT_SiO2 * (T - d_cts.Tref_SiO2)
            + x[:, 1] * d_cts.dVdT_Al2O3 * (T - d_cts.Tref_Al2O3)
            + x[:, 2] * d_cts.dVdT_Na2O * (T - d_cts.Tref_Na2O)
            + x[:, 3] * d_cts.dVdT_K2O * (T - d_cts.Tref_K2O)
            + x[:, 4] * d_cts.dVdT_MgO * (T - d_cts.Tref_MgO)
            + x[:, 5] * d_cts.dVdT_CaO * (T - d_cts.Tref_CaO)
        )

        d_g = self.density_glass(x)  # glass density
        v_g = XMW_tot / d_g  # glass volume
        # melt volume estimated from glass plus deviation from T ref
        v_l = v_g.reshape(-1, 1) + c_Vm_Tref.reshape(-1, 1) + alpha_.reshape(-1, 1)
        out = XMW_tot / v_l  # output melt density
        return torch.reshape(out, (out.shape[0], 1))

    def fragility(self, x):
        """melt fragility"""
        return self.predict_all(x)['fragility']

    def S_B1(self, x):
        """Sellmeir B1"""
        return self.predict_all(x)['S_B1']

    def S_B2(self, x):
        """Sellmeir B2"""
        return self.predict_all(x)['S_B2']

    def S_B3(self, x):
        """Sellmeir B3"""
        return self.predict_all(x)['S_B3']

    def S_C1(self, x):
        """Sellmeir C1, with proper scaling"""
        return self.predict_all(x)['S_C1']

    def S_C2(self, x):
        """Sellmeir C2, with proper scaling"""
        return self.predict_all(x)['S_C2']

    def S_C3(self, x):
        """Sellmeir C3, with proper scaling"""
        return self.predict_all(x)['S_C3']

    def elastic_modulus(self, x):
        """elastic modulus"""
        return self.predict_all(x)['elastic_modulus']

    def cte(self, x):
        """coefficient of thermal expansion"""
        return self.predict_all(x)['cte']

    def abbe(self, x):
        """Abbe number"""
        return self.predict_all(x)['abbe']

    def liquidus(self, x):
        """liquidus temperature, K"""
        return self.predict_all(x)['liquidus']

    def a_theory(self, x):
        """Theoretical high T viscosity limit

        see Le Losq et al. 2017, JNCS 463, page 184
        and references cited therein
        """

        # attempt with theoretical calculation
        vm_ = self.vm_glass(x)  # partial molar volumes

        # calculation of glass molar volume
        # careful with the units, we want m3/mol
        v_g = 1e-6 * (
            vm_[:, 0] * x[:, 0]
            + vm_[:, 1] * x[:, 1]  # sio2 + al2o3
            + vm_[:, 2] * x[:, 2]
            + vm_[:, 3] * x[:, 3]  # na2o + k2o
            + vm_[:, 4] * x[:, 4]
            + vm_[:, 5] * x[:, 5]
        )  # mgo + cao

        # calculation of theoretical A
        out = torch.log10(Avogadro * Planck / v_g)
        return torch.reshape(out, (out.shape[0], 1))

    def b_cg(self, x):
        """B in free volume (CG) equation"""
        return (
            0.5
            * (12.0 - self.a_cg(x))
            * (
                self.tg(x)
                - self.to_cg(x)
                + torch.sqrt(
                    (self.tg(x) - self.to_cg(x)) ** 2 + self.c_cg(x) * self.tg(x)
                )
            )
        )

    def b_tvf(self, x):
        """B in VFT equation"""
        return (12.0 - self.a_tvf(x)) * (self.tg(x) - self.c_tvf(x))

    def be(self, x):
        """Be term in Adam-Gibbs equation given Ae, Tg and Scong(Tg)"""
        return (12.0 - self.ae(x)) * (self.tg(x) * self.sctg(x))

    def ag(self, x, T):
        """viscosity from the Adam-Gibbs equation, given chemistry X and temperature T

        """
        return self.predict_all(x, T=T)['ag']

    def myega(self, x, T):
        """viscosity from the MYEGA equation, given entries X and temperature T

        """
        return self.predict_all(x, T=T)['myega']

    def am(self, x, T):
        """viscosity from the Avramov-Mitchell equation, given entries X and temperature T

        """
        return self.predict_all(x, T=T)['am']

    def cg(self, x, T):
        """free volume theory viscosity equation, given entries X and temperature T

        """
        return self.predict_all(x, T=T)['cg']

    def tvf(self, x, T):
        """Tamman-Vogel-Fulscher empirical viscosity, given entries X and temperature T

        need for speed = we decompose the calculation for making only one forward pass
        """
        return self.predict_all(x, T=T)['tvf']

    def sellmeier(self, x, lbd):
        """Sellmeier equation for refractive index calculation, with lbd in microns"""
        return self.predict_all(x, lbd=lbd)['sellmeier']

###
### TRAINING FUNCTIONS
###

class loss_scales:
    """loss scales for everything"""

    def __init__(self):
        # scaling coefficients for loss function
        # viscosity is always one
        self.visco = 1.0
        self.entro = 1.0
        self.raman = 20.0
        self.density = 1250.0
        self.ri = 5000.0
        self.tg = 0.001
        self.A_scale = 1e4  # 1e-6 potentiellement bien
        self.cpl = 1e-2  # 1e-2 for strong constraints

def training(
    neuralmodel,
    ds,
    criterion,
    optimizer,
    save_switch=True,
    save_name="./temp",
    nb_folds=1,
    train_patience=50,
    min_delta=0.1,
    verbose=True,
    mode="main",
    device="cuda",
):
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
    save_switch : bool
        if True, the network will be saved in save_name
    save_name : string
        the path to save the model during training
    nb_folds : int, default = 10
        the number of folds for the K-fold training
    train_patience : int, default = 50
        the number of iterations
    min_delta : float, default = 0.1
        Minimum decrease in the loss to qualify as an improvement,
        a decrease of less than or equal to `min_delta` will count as no improvement.
    verbose : bool, default = True
        Do you want details during training?
    device : string, default = "cuda"
        the device where the calculations are made during training

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

    # put model in train mode
    neuralmodel.train()

    # for early stopping
    epoch = 0
    best_epoch = 0
    val_ex = 0

    # for recording losses
    record_train_loss = []
    record_valid_loss = []

    # new vectors for the K-fold training (each vector contains slices of data separated)
    slices_x_visco_train = [ds.x_visco_train[i::nb_folds] for i in range(nb_folds)]
    slices_y_visco_train = [ds.y_visco_train[i::nb_folds] for i in range(nb_folds)]
    slices_T_visco_train = [ds.T_visco_train[i::nb_folds] for i in range(nb_folds)]

    slices_x_raman_train = [ds.x_raman_train[i::nb_folds] for i in range(nb_folds)]
    slices_y_raman_train = [ds.y_raman_train[i::nb_folds] for i in range(nb_folds)]

    slices_x_density_train = [ds.x_density_train[i::nb_folds] for i in range(nb_folds)]
    slices_y_density_train = [ds.y_density_train[i::nb_folds] for i in range(nb_folds)]

    slices_x_entro_train = [ds.x_entro_train[i::nb_folds] for i in range(nb_folds)]
    slices_y_entro_train = [ds.y_entro_train[i::nb_folds] for i in range(nb_folds)]

    slices_x_ri_train = [ds.x_ri_train[i::nb_folds] for i in range(nb_folds)]
    slices_y_ri_train = [ds.y_ri_train[i::nb_folds] for i in range(nb_folds)]
    slices_lbd_ri_train = [ds.lbd_ri_train[i::nb_folds] for i in range(nb_folds)]

    slices_x_cpl_train = [ds.x_cpl_train[i::nb_folds] for i in range(nb_folds)]
    slices_y_cpl_train = [ds.y_cpl_train[i::nb_folds] for i in range(nb_folds)]
    slices_T_cpl_train = [ds.T_cpl_train[i::nb_folds] for i in range(nb_folds)]

    slices_x_elastic_train = [ds.x_elastic_train[i::nb_folds] for i in range(nb_folds)]
    slices_y_elastic_train = [ds.y_elastic_train[i::nb_folds] for i in range(nb_folds)]

    slices_x_cte_train = [ds.x_cte_train[i::nb_folds] for i in range(nb_folds)]
    slices_y_cte_train = [ds.y_cte_train[i::nb_folds] for i in range(nb_folds)]

    slices_x_abbe_train = [ds.x_abbe_train[i::nb_folds] for i in range(nb_folds)]
    slices_y_abbe_train = [ds.y_abbe_train[i::nb_folds] for i in range(nb_folds)]

    slices_x_liquidus_train = [
        ds.x_liquidus_train[i::nb_folds] for i in range(nb_folds)
    ]
    slices_y_liquidus_train = [
        ds.y_liquidus_train[i::nb_folds] for i in range(nb_folds)
    ]

    while val_ex <= train_patience:

        #
        # TRAINING
        #
        loss = 0  # initialize the sum of losses of each fold

        for i in range(nb_folds):  # loop for K-Fold training to reduce memory footprint

            # training dataset is not on device yet and needs to be sent there
            x_visco_train = slices_x_visco_train[i].to(device)
            y_visco_train = slices_y_visco_train[i].to(device)
            T_visco_train = slices_T_visco_train[i].to(device)

            x_raman_train = slices_x_raman_train[i].to(device)
            y_raman_train = slices_y_raman_train[i].to(device)

            x_density_train = slices_x_density_train[i].to(device)
            y_density_train = slices_y_density_train[i].to(device)

            x_elastic_train = slices_x_elastic_train[i].to(device)
            y_elastic_train = slices_y_elastic_train[i].to(device)

            x_entro_train = slices_x_entro_train[i].to(device)
            y_entro_train = slices_y_entro_train[i].to(device)

            x_ri_train = slices_x_ri_train[i].to(device)
            y_ri_train = slices_y_ri_train[i].to(device)
            lbd_ri_train = slices_lbd_ri_train[i].to(device)

            x_cpl_train = slices_x_cpl_train[i].to(device)
            y_cpl_train = slices_y_cpl_train[i].to(device)
            T_cpl_train = slices_T_cpl_train[i].to(device)

            x_cte_train = slices_x_cte_train[i].to(device)
            y_cte_train = slices_y_cte_train[i].to(device)

            x_abbe_train = slices_x_abbe_train[i].to(device)
            y_abbe_train = slices_y_abbe_train[i].to(device)

            x_liquidus_train = slices_x_liquidus_train[i].to(device)
            y_liquidus_train = slices_y_liquidus_train[i].to(device)

            # Forward pass on training set

            # use i-Melt v2.2 API to speed up viscosity calculations
            visco_preds = neuralmodel.predict_all(x_visco_train,
                                                  T = T_visco_train)
            y_ag_pred_train = visco_preds["ag"]
            y_myega_pred_train = visco_preds["myega"]
            y_am_pred_train = visco_preds["am"]
            y_cg_pred_train = visco_preds["cg"]
            y_tvf_pred_train = visco_preds["tvf"]
            y_raman_pred_train = neuralmodel.raman_pred(x_raman_train)
            y_density_pred_train = neuralmodel.density_glass(x_density_train)
            y_entro_pred_train = neuralmodel.sctg(x_entro_train)
            y_ri_pred_train = neuralmodel.sellmeier(x_ri_train, lbd=lbd_ri_train)
            y_cpl_pred_train = neuralmodel.cpl(x_cpl_train, T_cpl_train)
            y_elastic_pred_train = neuralmodel.elastic_modulus(x_elastic_train)
            y_cte_pred_train = neuralmodel.cte(x_cte_train)
            y_abbe_pred_train = neuralmodel.abbe(x_abbe_train)
            y_liquidus_pred_train = neuralmodel.liquidus(x_liquidus_train)

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
            loss_raman = precision_raman * criterion(y_raman_pred_train, y_raman_train)
            loss_density = precision_density * criterion(
                y_density_pred_train, y_density_train
            )
            loss_entro = precision_entro * criterion(y_entro_pred_train, y_entro_train)
            loss_ri = precision_ri * criterion(y_ri_pred_train, y_ri_train)
            loss_cpl = precision_cpl * criterion(y_cpl_pred_train, y_cpl_train)
            loss_elastic = precision_elastic * criterion(
                y_elastic_pred_train, y_elastic_train
            )
            loss_cte = precision_cte * criterion(y_cte_pred_train, y_cte_train)
            loss_abbe = precision_abbe * criterion(y_abbe_pred_train, y_abbe_train)
            loss_liquidus = precision_liquidus * criterion(
                y_liquidus_pred_train, y_liquidus_train
            )

            loss_fold = (
                loss_ag
                + loss_myega
                + loss_am
                + loss_cg
                + loss_tvf
                + loss_raman
                + loss_density
                + loss_entro
                + loss_ri
                + loss_cpl
                + loss_elastic
                + loss_cte
                + loss_abbe
                + loss_liquidus
                + neuralmodel.log_vars[0]
                + neuralmodel.log_vars[1]
                + neuralmodel.log_vars[2]
                + neuralmodel.log_vars[3]
                + neuralmodel.log_vars[4]
                + neuralmodel.log_vars[5]
                + neuralmodel.log_vars[6]
                + neuralmodel.log_vars[7]
                + neuralmodel.log_vars[8]
                + neuralmodel.log_vars[9]
            )

            # initialise gradient
            optimizer.zero_grad()
            loss_fold.backward()  # backward gradient determination
            optimizer.step()  # optimiser call and step

            loss += loss_fold.item()  # add the new fold loss to the sum

        # record global loss (mean of the losses of the training folds)
        record_train_loss.append(loss / nb_folds)

        #
        # MONITORING VALIDATION SUBSET
        #
        with torch.set_grad_enabled(False):

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
            visco_preds = neuralmodel.predict_all(ds.x_visco_valid.to(device),
                                                  T = ds.T_visco_valid.to(device))
            y_ag_pred_valid = visco_preds["ag"]
            y_myega_pred_valid = visco_preds["myega"]
            y_am_pred_valid = visco_preds["am"]
            y_cg_pred_valid = visco_preds["cg"]
            y_tvf_pred_valid = visco_preds["tvf"]

            y_raman_pred_valid = neuralmodel.raman_pred(ds.x_raman_valid.to(device))
            y_density_pred_valid = neuralmodel.density_glass(
                ds.x_density_valid.to(device)
            )
            y_entro_pred_valid = neuralmodel.sctg(ds.x_entro_valid.to(device))
            y_ri_pred_valid = neuralmodel.sellmeier(
                ds.x_ri_valid.to(device), lbd=ds.lbd_ri_valid.to(device)
            )
            y_clp_pred_valid = neuralmodel.cpl(
                ds.x_cpl_valid.to(device), ds.T_cpl_valid.to(device)
            )
            y_elastic_pred_valid = neuralmodel.elastic_modulus(
                ds.x_elastic_valid.to(device)
            )
            y_cte_pred_valid = neuralmodel.cte(ds.x_cte_valid.to(device))
            y_abbe_pred_valid = neuralmodel.abbe(ds.x_abbe_valid.to(device))
            y_liquidus_pred_valid = neuralmodel.liquidus(ds.x_liquidus_valid.to(device))

            # validation loss
            loss_ag_v = precision_visco * criterion(
                y_ag_pred_valid, ds.y_visco_valid.to(device)
            )
            loss_myega_v = precision_visco * criterion(
                y_myega_pred_valid, ds.y_visco_valid.to(device)
            )
            loss_am_v = precision_visco * criterion(
                y_am_pred_valid, ds.y_visco_valid.to(device)
            )
            loss_cg_v = precision_visco * criterion(
                y_cg_pred_valid, ds.y_visco_valid.to(device)
            )
            loss_tvf_v = precision_visco * criterion(
                y_tvf_pred_valid, ds.y_visco_valid.to(device)
            )
            loss_raman_v = precision_raman * criterion(
                y_raman_pred_valid, ds.y_raman_valid.to(device)
            )
            loss_density_v = precision_density * criterion(
                y_density_pred_valid, ds.y_density_valid.to(device)
            )
            loss_entro_v = precision_entro * criterion(
                y_entro_pred_valid, ds.y_entro_valid.to(device)
            )
            loss_ri_v = precision_ri * criterion(
                y_ri_pred_valid, ds.y_ri_valid.to(device)
            )
            loss_cpl_v = precision_cpl * criterion(
                y_clp_pred_valid, ds.y_cpl_valid.to(device)
            )
            loss_elastic_v = precision_elastic * criterion(
                y_elastic_pred_valid, ds.y_elastic_valid.to(device)
            )
            loss_cte_v = precision_cte * criterion(
                y_cte_pred_valid, ds.y_cte_valid.to(device)
            )
            loss_abbe_v = precision_abbe * criterion(
                y_abbe_pred_valid, ds.y_abbe_valid.to(device)
            )
            loss_liquidus_v = precision_liquidus * criterion(
                y_liquidus_pred_valid, ds.y_liquidus_valid.to(device)
            )

            loss_v = (
                loss_ag_v
                + loss_myega_v
                + loss_am_v
                + loss_cg_v
                + loss_tvf_v
                + loss_raman_v
                + loss_density_v
                + loss_entro_v
                + loss_ri_v
                + loss_cpl_v
                + loss_elastic_v
                + loss_cte_v
                + loss_abbe_v
                + loss_liquidus_v
                + neuralmodel.log_vars[0]
                + neuralmodel.log_vars[1]
                + neuralmodel.log_vars[2]
                + neuralmodel.log_vars[3]
                + neuralmodel.log_vars[4]
                + neuralmodel.log_vars[5]
                + neuralmodel.log_vars[6]
                + neuralmodel.log_vars[7]
                + neuralmodel.log_vars[8]
                + neuralmodel.log_vars[9]
            )

            record_valid_loss.append(loss_v.item())

        #
        # Print info on screen
        #
        if verbose == True:
            if epoch % 100 == 0:
                print(
                    "\nTRAIN -- Raman: {:.3f}, d: {:.3f}, S: {:.3f}, RI: {:.3f}, V: {:.3f}, Cp: {:.3f}, Em: {:.3f}, CTE: {:.3f}, Ab: {:.3f}, Tl: {:.3f}".format(
                        loss_raman,
                        loss_density,
                        loss_entro,
                        loss_ri,
                        loss_ag,
                        loss_cpl,
                        loss_elastic,
                        loss_cte,
                        loss_abbe,
                        loss_liquidus,
                    )
                )
                print(
                    "VALID -- Raman: {:.3f}, d: {:.3f}, S: {:.3f}, RI: {:.3f}, V: {:.3f}, Cp: {:.3f}, Em: {:.3f}, CTE: {:.3f}, Ab: {:.3f}, Tl: {:.3f}\n".format(
                        loss_raman_v,
                        loss_density_v,
                        loss_entro_v,
                        loss_ri_v,
                        loss_ag_v,
                        loss_cpl_v,
                        loss_elastic_v,
                        loss_cte_v,
                        loss_abbe_v,
                        loss_liquidus_v,
                    )
                )
            if epoch % 20 == 0:
                print(
                    "Epoch {} => loss train {:.2f}, valid {:.2f}; reg A: {:.6f}".format(
                        epoch, loss / nb_folds, loss_v.item(), 0
                    )
                )

        #
        # calculating ES criterion
        #
        if epoch == 0:
            val_ex = 0
            best_loss_v = loss_v.item()
        elif (
            loss_v.item() <= best_loss_v - min_delta
        ):  # if improvement is significant, this saves the model
            val_ex = 0
            best_epoch = epoch
            best_loss_v = loss_v.item()

            if save_switch == True:  # save best model
                torch.save(neuralmodel.state_dict(), save_name)
        else:
            val_ex += 1

        epoch += 1

    # print outputs if verbose is True
    if verbose == True:
        time2 = time.time()
        print("Running time in seconds:", time2 - time1)
        print("Scaled loss values are:")
        print(
            "\nTRAIN -- Raman: {:.3f}, d: {:.3f}, S: {:.3f}, RI: {:.3f}, V: {:.3f}, Cp: {:.3f}, Em: {:.3f}, CTE: {:.3f}, Ab: {:.3f}, Tl: {:.3f}".format(
                loss_raman,
                loss_density,
                loss_entro,
                loss_ri,
                loss_ag,
                loss_cpl,
                loss_elastic,
                loss_cte,
                loss_abbe,
                loss_liquidus,
            )
        )
        print(
            "VALID -- Raman: {:.3f}, d: {:.3f}, S: {:.3f}, RI: {:.3f}, V: {:.3f}, Cp: {:.3f}, Em: {:.3f}, CTE: {:.3f}, Ab: {:.3f}, Tl: {:.3f}\n".format(
                loss_raman_v,
                loss_density_v,
                loss_entro_v,
                loss_ri_v,
                loss_ag_v,
                loss_cpl_v,
                loss_elastic_v,
                loss_cte_v,
                loss_abbe_v,
                loss_liquidus_v,
            )
        )

    return neuralmodel, record_train_loss, record_valid_loss

def record_loss_build(path, list_models, ds):
    """build a Pandas dataframe with the losses for a list of models at path"""
    # scaling coefficients for global loss function
    # viscosity is always one
    # check lines 578-582 in imelt.py
    entro_scale = 1.0
    raman_scale = 20.0
    density_scale = 1000.0
    ri_scale = 10000.0

    nb_exp = len(list_models)

    record_loss = pd.DataFrame()

    record_loss["name"] = list_models

    record_loss["nb_layers"] = np.zeros(nb_exp)
    record_loss["nb_neurons"] = np.zeros(nb_exp)
    record_loss["p_drop"] = np.zeros(nb_exp)

    record_loss["loss_ag_train"] = np.zeros(nb_exp)
    record_loss["loss_ag_valid"] = np.zeros(nb_exp)

    record_loss["loss_am_train"] = np.zeros(nb_exp)
    record_loss["loss_am_valid"] = np.zeros(nb_exp)

    record_loss["loss_am_train"] = np.zeros(nb_exp)
    record_loss["loss_am_valid"] = np.zeros(nb_exp)

    record_loss["loss_Sconf_train"] = np.zeros(nb_exp)
    record_loss["loss_Sconf_valid"] = np.zeros(nb_exp)

    record_loss["loss_d_train"] = np.zeros(nb_exp)
    record_loss["loss_d_valid"] = np.zeros(nb_exp)

    record_loss["loss_raman_train"] = np.zeros(nb_exp)
    record_loss["loss_raman_valid"] = np.zeros(nb_exp)

    record_loss["loss_train"] = np.zeros(nb_exp)
    record_loss["loss_valid"] = np.zeros(nb_exp)

    # Loss criterion
    criterion = torch.nn.MSELoss()

    # Load dataset
    for idx, name in enumerate(list_models):

        # Extract arch
        nb_layers = int(name[name.find("l") + 1 : name.find("_")])
        nb_neurons = int(name[name.find("n") + 1 : name.find("p") - 1])
        p_drop = float(name[name.find("p") + 1 : name.find("s") - 1])

        # Record arch
        record_loss.loc[idx, "nb_layers"] = nb_layers
        record_loss.loc[idx, "nb_neurons"] = nb_neurons
        record_loss.loc[idx, "p_drop"] = p_drop

        # Declare model
        neuralmodel = imelt.model(
            6, nb_neurons, nb_layers, ds.nb_channels_raman, p_drop=p_drop
        )
        neuralmodel.load_state_dict(torch.load(path + "/" + name, map_location="cpu"))
        neuralmodel.eval()

        # PREDICTIONS

        with torch.set_grad_enabled(False):
            # train
            y_ag_pred_train = neuralmodel.ag(ds.x_visco_train, ds.T_visco_train)
            y_myega_pred_train = neuralmodel.myega(ds.x_visco_train, ds.T_visco_train)
            y_am_pred_train = neuralmodel.am(ds.x_visco_train, ds.T_visco_train)
            y_cg_pred_train = neuralmodel.cg(ds.x_visco_train, ds.T_visco_train)
            y_tvf_pred_train = neuralmodel.tvf(ds.x_visco_train, ds.T_visco_train)
            y_raman_pred_train = neuralmodel.raman_pred(ds.x_raman_train)
            y_density_pred_train = neuralmodel.density_glass(ds.x_density_train)
            y_entro_pred_train = neuralmodel.sctg(ds.x_entro_train)
            y_ri_pred_train = neuralmodel.sellmeier(ds.x_ri_train, ds.lbd_ri_train)

            # valid
            y_ag_pred_valid = neuralmodel.ag(ds.x_visco_valid, ds.T_visco_valid)
            y_myega_pred_valid = neuralmodel.myega(ds.x_visco_valid, ds.T_visco_valid)
            y_am_pred_valid = neuralmodel.am(ds.x_visco_valid, ds.T_visco_valid)
            y_cg_pred_valid = neuralmodel.cg(ds.x_visco_valid, ds.T_visco_valid)
            y_tvf_pred_valid = neuralmodel.tvf(ds.x_visco_valid, ds.T_visco_valid)
            y_raman_pred_valid = neuralmodel.raman_pred(ds.x_raman_valid)
            y_density_pred_valid = neuralmodel.density_glass(ds.x_density_valid)
            y_entro_pred_valid = neuralmodel.sctg(ds.x_entro_valid)
            y_ri_pred_valid = neuralmodel.sellmeier(ds.x_ri_valid, ds.lbd_ri_valid)

            # Compute Loss

            # train
            record_loss.loc[idx, "loss_ag_train"] = np.sqrt(
                criterion(y_ag_pred_train, ds.y_visco_train).item()
            )
            record_loss.loc[idx, "loss_myega_train"] = np.sqrt(
                criterion(y_myega_pred_train, ds.y_visco_train).item()
            )
            record_loss.loc[idx, "loss_am_train"] = np.sqrt(
                criterion(y_am_pred_train, ds.y_visco_train).item()
            )
            record_loss.loc[idx, "loss_cg_train"] = np.sqrt(
                criterion(y_cg_pred_train, ds.y_visco_train).item()
            )
            record_loss.loc[idx, "loss_tvf_train"] = np.sqrt(
                criterion(y_tvf_pred_train, ds.y_visco_train).item()
            )
            record_loss.loc[idx, "loss_raman_train"] = np.sqrt(
                criterion(y_raman_pred_train, ds.y_raman_train).item()
            )
            record_loss.loc[idx, "loss_d_train"] = np.sqrt(
                criterion(y_density_pred_train, ds.y_density_train).item()
            )
            record_loss.loc[idx, "loss_Sconf_train"] = np.sqrt(
                criterion(y_entro_pred_train, ds.y_entro_train).item()
            )
            record_loss.loc[idx, "loss_ri_train"] = np.sqrt(
                criterion(y_ri_pred_train, ds.y_ri_train).item()
            )

            # validation
            record_loss.loc[idx, "loss_ag_valid"] = np.sqrt(
                criterion(y_ag_pred_valid, ds.y_visco_valid).item()
            )
            record_loss.loc[idx, "loss_myega_valid"] = np.sqrt(
                criterion(y_myega_pred_valid, ds.y_visco_valid).item()
            )
            record_loss.loc[idx, "loss_am_valid"] = np.sqrt(
                criterion(y_am_pred_valid, ds.y_visco_valid).item()
            )
            record_loss.loc[idx, "loss_cg_valid"] = np.sqrt(
                criterion(y_cg_pred_valid, ds.y_visco_valid).item()
            )
            record_loss.loc[idx, "loss_tvf_valid"] = np.sqrt(
                criterion(y_tvf_pred_valid, ds.y_visco_valid).item()
            )
            record_loss.loc[idx, "loss_raman_valid"] = np.sqrt(
                criterion(y_raman_pred_valid, ds.y_raman_valid).item()
            )
            record_loss.loc[idx, "loss_d_valid"] = np.sqrt(
                criterion(y_density_pred_valid, ds.y_density_valid).item()
            )
            record_loss.loc[idx, "loss_Sconf_valid"] = np.sqrt(
                criterion(y_entro_pred_valid, ds.y_entro_valid).item()
            )
            record_loss.loc[idx, "loss_ri_valid"] = np.sqrt(
                criterion(y_ri_pred_valid, ds.y_ri_valid).item()
            )

            record_loss.loc[idx, "loss_train"] = (
                record_loss.loc[idx, "loss_ag_train"]
                + record_loss.loc[idx, "loss_myega_train"]
                + record_loss.loc[idx, "loss_am_train"]
                + record_loss.loc[idx, "loss_cg_train"]
                + record_loss.loc[idx, "loss_tvf_train"]
                + raman_scale * record_loss.loc[idx, "loss_raman_train"]
                + density_scale * record_loss.loc[idx, "loss_d_train"]
                + entro_scale * record_loss.loc[idx, "loss_Sconf_train"]
                + ri_scale * record_loss.loc[idx, "loss_ri_train"]
            )

            record_loss.loc[idx, "loss_valid"] = (
                record_loss.loc[idx, "loss_ag_valid"]
                + record_loss.loc[idx, "loss_myega_valid"]
                + record_loss.loc[idx, "loss_am_valid"]
                + record_loss.loc[idx, "loss_cg_valid"]
                + record_loss.loc[idx, "loss_tvf_valid"]
                + raman_scale * record_loss.loc[idx, "loss_raman_valid"]
                + density_scale * record_loss.loc[idx, "loss_d_valid"]
                + entro_scale * record_loss.loc[idx, "loss_Sconf_valid"]
                + ri_scale * record_loss.loc[idx, "loss_ri_valid"]
            )

    return record_loss

###
### BAGGING PREDICTOR
###
class bagging_models:
    """custom class for bagging models and making predictions

    Parameters
    ----------
    path : str
        path of models

    name_models : list of str
        names of models

    device : str
        cpu or gpu

    activation_function : torch.nn.Module
        activation function to be used, default is ReLU

    Methods
    -------
    predict : function
        make predictions

    """

    def __init__(
        self, path, name_models, ds, device, activation_function=torch.nn.GELU()
    ):

        self.device = device
        self.n_models = len(name_models)
        self.models = [None for _ in range(self.n_models)]

        for i in range(self.n_models):
            name = name_models[i]

            # Extract arch
            nb_layers = int(name[name.find("l") + 1 : name.find("_n")])
            nb_neurons = int(name[name.find("n") + 1 : name.rfind("_p")])
            p_drop = float(name[name.find("p") + 1 : name.rfind("_m")])

            self.models[i] = imelt.model(
                ds.x_visco_train.shape[1],
                nb_neurons,
                nb_layers,
                ds.nb_channels_raman,
                p_drop=p_drop,
                activation_function=activation_function,
            )

            state_dict = torch.load(path / name, map_location="cpu")
            if len(state_dict) == 2:
                self.models[i].load_state_dict(state_dict[0])
            else:
                self.models[i].load_state_dict(state_dict)
            self.models[i].eval()

    def predict(
        self,
        methods,
        X,
        T = np.array([1000.0]),
        lbd = np.array([589.0*1e-3]), # in micrometers !
        sampling = False,
        n_sample = 10,
    ):
        """returns predictions from the n models

        Parameters
        ----------
        methods : list
            list of the properties to predict. Choose between:
        X : pandas dataframe
            chemical composition for prediction
        T : 1d numpy array or pandas dataframe
            temperatures for predictions, default = np.array([1000.0,])
        lbd : 1d numpy array or pandas dataframe
            wavelength in micrometer for Sellmeier equation, default = np.array([589.0*1e-3])
        sampling : Bool
            if True, dropout is activated and n_sample random samples will be generated per network.
            This allows performing MC Dropout on the ensemble of models.
        n_sample : Int, optional
        """

        # to ensure compatibility with older versions of imelt
        if isinstance(methods, str):
            methods = [methods]
            methods_oldAPI = True
            warnings.warn("methods is a string, using old API. Please use a list of methods in the future instead.", DeprecationWarning)
        else:
            methods_oldAPI = False

        # Handle different input types for X and send to device
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X).to(self.device)  # Convert to FloatTensor
        elif isinstance(X, pd.DataFrame):
            X = torch.FloatTensor(X.values).to(
                self.device
            )  # Extract values and convert to Tensor

        if isinstance(T, np.ndarray):
            T = torch.Tensor(T.reshape(-1, 1)).to(self.device)
        elif isinstance(T, pd.DataFrame):
            T = torch.Tensor(T.values.reshape(-1, 1)).to(self.device)
        if isinstance(lbd, np.ndarray):
            lbd = torch.Tensor(lbd.reshape(-1, 1)).to(self.device)
        elif isinstance(lbd, pd.DataFrame):
            lbd = torch.Tensor(lbd.values.reshape(-1, 1)).to(self.device)

        # Handle cases where one variable has a single value and others have multiple
        if len(T) > 1 and len(X) == 1:
            X = torch.tile(X, (len(T), 1))
        elif len(X) > 1 and len(T) == 1:
            T = torch.tile(T, (len(X), 1))
        if len(lbd) > 1 and len(X) == 1:
            X = torch.tile(X, (len(lbd), 1))
        elif len(X) > 1 and len(lbd) == 1:
            lbd = torch.tile(lbd, (len(X), 1))

        # if we don't sample, we only need one sample
        if sampling == False:
            n_sample = 1

        # take care of model device and MC dropout        
        for i in range(self.n_models):
            self.models[i].to(self.device) # send model to device
            if sampling == True:
                self.models[i].train() # activate dropout for MC sampling
            
        # save output dictionaries from predict_all() in a list
        # preallocation
        out_dict_models = [None for i in range(self.n_models*n_sample)]

        # make predictions
        with torch.no_grad():
            count = 0
            for i in range(self.n_models):
                for j in range(n_sample):
                    out_dict_models[count] = self.models[i].predict_all(X, T=T, lbd=lbd)
                    count += 1

                # if sampling, deactivate dropout before switching to the next model / leaving
                if sampling == True:
                    self.models[i].eval()
        
        # prepare output dictionary for the user
        output_dictionary = {}
        for method in methods:
            # Handling specific cases first, then generalities
            if method == "raman_spectra":
                out = np.zeros(
                    (len(X), 850, self.n_models*n_sample)
                )  # problem is defined with a X raman shift of 850 values
                for i in range(self.n_models*n_sample):
                        out[:, :, i] = (
                            out_dict_models[i][method]
                            .cpu()
                            .detach()
                            .numpy()
                        )   
                    # out = out.reshape(
                    #     (out.shape[0], out.shape[1], out.shape[2] * out.shape[3])
                    # )

            elif method == "vm_glass":
                out = np.zeros((len(X), 6, self.n_models*n_sample))
                for i in range(self.n_models*n_sample):
                        out[:, :, i] = (
                            out_dict_models[i][method]
                            .cpu()
                            .detach()
                            .numpy()
                        )  
            elif method == "partial_cpl":
                out = np.zeros((len(X), 8, self.n_models*n_sample))
                for i in range(self.n_models*n_sample):
                        out[:, :, i] = (
                            out_dict_models[i][method]
                            .cpu()
                            .detach()
                            .numpy()
                        )  
            else:
                out = np.zeros((len(X), self.n_models*n_sample))
                for i in range(self.n_models*n_sample):
                        out[:, i] = (
                            out_dict_models[i][method]
                            .cpu()
                            .detach()
                            .numpy()
                            .reshape(-1)
                        )  

            # final storage
            if sampling == False:
                output_dictionary[method] = np.median(out, axis=out.ndim - 1)
            else:
                output_dictionary[method] = out
        
        # to ensure compatibility with older versions of imelt
        if methods_oldAPI == True:
            # if the user used the old API, we return a single array
            output_dictionary = output_dictionary[methods[0]]

        return output_dictionary

def load_pretrained_bagged(
    device=torch.device("cpu"), activation_function=torch.nn.GELU()
):
    """loader for the pretrained bagged i-melt models

    Parameters
    ----------
    device : torch.device()
        CPU or GPU device. default : 'cpu' (optional)
    activation_function : torch activation function
        activation function of the networks. default : torch.nn.GELU()

    Returns
    -------
    bagging_models : object
        A bagging_models object that can be used for predictions
    """

    # we use the default paths in imelt.data_loader()
    ds = imelt.data_loader()

    # we get the list of model names
    name_list = pd.read_csv(_BASEMODELPATH / "best_list.csv").loc[:, "name"]
    return bagging_models(
        _BASEMODELPATH, name_list, ds, device, activation_function=activation_function
    )


class constants:
    def __init__(self):
        self.V_g_sio2 = (27 + 2 * 16) / 2.2007
        self.V_g_al2o3 = (26 * 2 + 3 * 16) / 3.009
        self.V_g_na2o = (22 * 2 + 16) / 2.686
        self.V_g_k2o = (44 * 2 + 16) / 2.707
        self.V_g_mgo = (24.3 + 16) / 3.115
        self.V_g_cao = (40.08 + 16) / 3.140

        self.V_m_sio2 = 27.297  # Courtial and Dingwell 1999, 1873 K
        self.V_m_al2o3 = 36.666  # Courtial and Dingwell 1999
        # self.V_m_SiCa = -7.105 # Courtial and Dingwell 1999
        self.V_m_na2o = 29.65  # Tref=1773 K (Lange, 1997; CMP)
        self.V_m_k2o = 47.28  # Tref=1773 K (Lange, 1997; CMP)
        self.V_m_mgo = 12.662  # Courtial and Dingwell 1999
        self.V_m_cao = 20.664  # Courtial and Dingwell 1999

        # dV/dT values
        self.dVdT_SiO2 = 1.157e-3  # Courtial and Dingwell 1999
        self.dVdT_Al2O3 = -1.184e-3  # Courtial and Dingwell 1999
        # self.dVdT_SiCa = -2.138 # Courtial and Dingwell 1999
        self.dVdT_Na2O = 0.00768  # Table 4 (Lange, 1997)
        self.dVdT_K2O = 0.01208  # Table 4 (Lange, 1997)
        self.dVdT_MgO = 1.041e-3  # Courtial and Dingwell 1999
        self.dVdT_CaO = 3.786e-3  # Courtial and Dingwell 1999

        # melt T reference
        self.Tref_SiO2 = 1873.0  # Courtial and Dingwell 1999
        self.Tref_Al2O3 = 1873.0  # Courtial and Dingwell 1999
        self.Tref_Na2O = 1773.0  # Tref=1773 K (Lange, 1997; CMP)
        self.Tref_K2O = 1773.0  # Tref=1773 K (Lange, 1997; CMP)
        self.Tref_MgO = 1873.0  # Courtial and Dingwell 1999
        self.Tref_CaO = 1873.0  # Courtial and Dingwell 1999

        # correction constants between glass at Tambient and melt at Tref
        self.c_sio2 = self.V_m_sio2 - self.V_g_sio2
        self.c_al2o3 = self.V_m_al2o3 - self.V_g_al2o3
        self.c_na2o = self.V_m_na2o - self.V_g_na2o
        self.c_k2o = self.V_m_k2o - self.V_g_k2o
        self.c_mgo = self.V_m_mgo - self.V_g_mgo
        self.c_cao = self.V_m_cao - self.V_g_cao


class density_constants:

    def __init__(self):
        # Partial Molar Volumes
        self.MV_SiO2 = 27.297  # Courtial and Dingwell 1999
        self.MV_TiO2 = 28.32  # TiO2 at Tref=1773 K (Lange and Carmichael, 1987)
        self.MV_Al2O3 = 36.666  # Courtial and Dingwell 1999
        self.MV_Fe2O3 = 41.50  # Fe2O3 at Tref=1723 K (Liu and Lange, 2006)
        self.MV_FeO = 12.68  # FeO at Tref=1723 K (Guo et al., 2014)
        self.MV_MgO = 12.662  # Courtial and Dingwell 1999
        self.MV_CaO = 20.664  # Courtial and Dingwell 1999
        self.MV_SiCa = -7.105  # Courtial and Dingwell 1999
        self.MV_Na2O = 29.65  # Tref=1773 K (Lange, 1997; CMP)
        self.MV_K2O = 47.28  # Tref=1773 K (Lange, 1997; CMP)
        self.MV_H2O = 22.9  # H2O at Tref=1273 K (Ochs and Lange, 1999)

        # Partial Molar Volume uncertainties
        # value = 0 if not reported
        self.unc_MV_SiO2 = 0.152  # Courtial and Dingwell 1999
        self.unc_MV_TiO2 = 0.0
        self.unc_MV_Al2O3 = 0.196  # Courtial and Dingwell 1999
        self.unc_MV_Fe2O3 = 0.0
        self.unc_MV_FeO = 0.0
        self.unc_MV_MgO = 0.181  # Courtial and Dingwell 1999
        self.unc_MV_CaO = 0.123  # Courtial and Dingwell 1999
        self.unc_MV_SiCa = 0.509  # Courtial and Dingwell 1999
        self.unc_MV_Na2O = 0.07
        self.unc_MV_K2O = 0.10
        self.unc_MV_H2O = 0.60

        # dV/dT values
        # MgO, CaO, Na2O, K2O Table 4 (Lange, 1997)
        # SiO2, TiO2, Al2O3 Table 9 (Lange and Carmichael, 1987)
        # H2O from Ochs & Lange (1999)
        # Fe2O3 from Liu & Lange (2006)
        # FeO from Guo et al (2014)
        self.dVdT_SiO2 = 1.157e-3  # Courtial and Dingwell 1999
        self.dVdT_TiO2 = 0.00724
        self.dVdT_Al2O3 = -1.184e-3  # Courtial and Dingwell 1999
        self.dVdT_Fe2O3 = 0.0
        self.dVdT_FeO = 0.00369
        self.dVdT_MgO = 1.041e-3  # Courtial and Dingwell 1999
        self.dVdT_CaO = 3.786e-3  # Courtial and Dingwell 1999
        self.dVdT_SiCa = -2.138  # Courtial and Dingwell 1999
        self.dVdT_Na2O = 0.00768
        self.dVdT_K2O = 0.01208
        self.dVdT_H2O = 0.0095

        # dV/dT uncertainties
        # value = 0 if not reported
        self.unc_dVdT_SiO2 = 0.0007e-3  # Courtial and Dingwell 1999
        self.unc_dVdT_TiO2 = 0.0
        self.unc_dVdT_Al2O3 = 0.0009e-3  # Courtial and Dingwell 1999
        self.unc_dVdT_Fe2O3 = 0.0
        self.unc_dVdT_FeO = 0.0
        self.unc_dVdT_MgO = 0.0008  # Courtial and Dingwell 1999
        self.unc_dVdT_CaO = 0.0005e-3  # Courtial and Dingwell 1999
        self.unc_dVdT_SiCa = 0.002e-3  # Courtial and Dingwell 1999
        self.unc_dVdT_Na2O = 0.0
        self.unc_dVdT_K2O = 0.0
        self.unc_dVdT_H2O = 0.0008

        # dV/dP values
        # Anhydrous component data from Kess and Carmichael (1991)
        # H2O data from Ochs & Lange (1999)
        self.dVdP_SiO2 = -0.000189
        self.dVdP_TiO2 = -0.000231
        self.dVdP_Al2O3 = -0.000226
        self.dVdP_Fe2O3 = -0.000253
        self.dVdP_FeO = -0.000045
        self.dVdP_MgO = 0.000027
        self.dVdP_CaO = 0.000034
        self.dVdP_Na2O = -0.00024
        self.dVdP_K2O = -0.000675
        self.dVdP_H2O = -0.00032

        # dV/dP uncertainties
        self.unc_dVdP_SiO2 = 0.000002
        self.unc_dVdP_TiO2 = 0.000006
        self.unc_dVdP_Al2O3 = 0.000009
        self.unc_dVdP_Fe2O3 = 0.000009
        self.unc_dVdP_FeO = 0.000003
        self.unc_dVdP_MgO = 0.000007
        self.unc_dVdP_CaO = 0.000005
        self.unc_dVdP_Na2O = 0.000005
        self.unc_dVdP_K2O = 0.000014
        self.unc_dVdP_H2O = 0.000060

        # Tref values
        self.Tref_SiO2 = 1873.0  # Courtial and Dingwell 1999
        self.Tref_TiO2 = 1773.0
        self.Tref_Al2O3 = 1873.0  # Courtial and Dingwell 1999
        self.Tref_Fe2O3 = 1723.0
        self.Tref_FeO = 1723.0
        self.Tref_MgO = 1873.0  # Courtial and Dingwell 1999
        self.Tref_CaO = 1873.0  # Courtial and Dingwell 1999
        self.Tref_Na2O = 1773.0
        self.Tref_K2O = 1773.0
        self.Tref_H2O = 1273.0
