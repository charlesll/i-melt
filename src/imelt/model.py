# (c) Charles Le Losq et co 2022-2024
# see embedded licence file
# imelt V2.1

import numpy as np
import torch, time
import pandas as pd
from scipy.constants import Avogadro, Planck
import imelt as imelt

# to load data and models in a library
from pathlib import Path
import os

_BASEMODELPATH = Path(os.path.dirname(__file__)) / "models"
_BASEDATAPATH = Path(os.path.dirname(__file__)) / "data"


###
### MODEL
###
class PositionalEncoder(torch.nn.Module):
    """
    From:
    https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e
    Adapted from:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    https://github.com/LiamMaclean216/Pytorch-Transfomer/blob/master/utils.py
    """

    def __init__(
        self, dropout: float = 0.1, max_seq_len: int = 5000, d_model: int = 512
    ):
        """
        Args:
            dropout: the dropout rate
            max_seq_len: the maximum length of the input sequences
            d_model: The dimension of the output of sub-layers in the model
                     (Vaswani et al, 2017)
        """

        super().__init__()

        self.d_model = d_model

        self.dropout = torch.nn.Dropout(p=dropout)

        # Create constant positional encoding matrix with values
        # dependent on position and i
        position = torch.arange(max_seq_len).unsqueeze(1)

        exp_input = torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)

        div_term = torch.exp(
            exp_input
        )  # Returns a new tensor with the exponential of the elements of exp_input

        pe = torch.zeros(max_seq_len, d_model)

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(
            position * div_term
        )  # torch.Size([target_seq_len, dim_val])

        pe = pe.unsqueeze(0).transpose(
            0, 1
        )  # torch.Size([target_seq_len, input_size, dim_val])

        # register that pe is not a model parameter
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, enc_seq_len, dim_val]
        """

        add = self.pe[: x.size(1), :].squeeze(1)

        x = x + add

        return self.dropout(x)


class model(torch.nn.Module):
    """i-MELT model"""

    def __init__(
        self,
        input_size,
        hidden_size=300,
        num_layers=4,
        nb_channels_raman=800,
        p_drop=0.2,
        activation_function=torch.nn.ReLU(),
        shape="rectangle",
        dropout_pos_enc=0.01,
        n_heads=4,
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
            activation function for the hidden units, default = torch.nn.ReLU()
            choose here : https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity

        shape : string (optional)
            either a rectangle network (same number of neurons per layer, or triangle (regularly decreasing number of neurons per layer))
            default = rectangle

        dropout_pos_enc & n_heads are experimental features, do not use...
        """
        super(model, self).__init__()

        # init parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nb_channels_raman = nb_channels_raman
        self.shape = shape

        # get constants
        # self.constants = constants()

        # network related torch stuffs
        self.activation_function = activation_function
        self.p_drop = p_drop
        self.dropout = torch.nn.Dropout(p=p_drop)

        # for transformer
        self.dropout_pos_enc = dropout_pos_enc
        self.n_heads = n_heads

        # general shape of the network
        if self.shape == "rectangle":

            self.linears = torch.nn.ModuleList(
                [torch.nn.Linear(input_size, self.hidden_size)]
            )
            self.linears.extend(
                [
                    torch.nn.Linear(self.hidden_size, self.hidden_size)
                    for i in range(1, self.num_layers)
                ]
            )

        if self.shape == "triangle":

            self.linears = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        self.input_size, int(self.hidden_size / self.num_layers)
                    )
                ]
            )
            self.linears.extend(
                [
                    torch.nn.Linear(
                        int(self.hidden_size / self.num_layers * i),
                        int(self.hidden_size / self.num_layers * (i + 1)),
                    )
                    for i in range(1, self.num_layers)
                ]
            )
        if self.shape == "transformer":

            # Creating the three linear layers needed for the model
            self.encoder_input_layer = torch.nn.Linear(
                in_features=1, out_features=self.hidden_size
            )

            # Create positional encoder
            self.positional_encoding_layer = PositionalEncoder(
                d_model=self.hidden_size,
                dropout=self.dropout_pos_enc,
                max_seq_len=self.hidden_size,
            )

            # The encoder layer used in the paper is identical to the one used by
            # Vaswani et al (2017) on which the PyTorch module is based.
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=self.n_heads,
                dropout=self.p_drop,
                batch_first=True,
            )

            # Stack the encoder layers in nn.TransformerDecoder
            # It seems the option of passing a normalization instance is redundant
            # in my case, because nn.TransformerEncoderLayer per default normalizes
            # after each sub-layer
            # (https://github.com/pytorch/pytorch/issues/24930).
            self.encoder = torch.nn.TransformerEncoder(
                encoder_layer=encoder_layer, num_layers=self.num_layers, norm=None
            )

        ###
        # output layers
        ###
        if self.shape == "transformer":
            self.out_thermo = torch.nn.Linear(
                self.hidden_size * self.input_size, 34
            )  # Linear output, 22 without Cp
            self.out_raman = torch.nn.Linear(
                self.hidden_size * self.input_size, self.nb_channels_raman
            )  # Linear output
        else:
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

        # we are going to determine better values for
        # the aCpl and bCpl coefficients
        # self.Cp_coefs = torch.nn.Parameter(
        #     torch.tensor(
        #     [np.log(81.37), np.log(85.78), np.log(100.6), np.log(50.13), np.log(85.78), np.log(86.05),
        #      np.log(0.09428), np.log(0.01578)]),
        #      requires_grad=True
        #     )

        # here are parameters if we try minimizing the m vs Cpconf/Sconf relationship
        # self.m_cp_s_coefs = torch.nn.Parameter(
        #     torch.tensor([np.log(10.), np.log(15.)]),
        #     requires_grad=True)

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
        if self.shape != "transformer":
            for layer in self.linears:  # Feedforward
                x = self.dropout(self.activation_function(layer(x)))
            return x
        else:
            x = self.encoder_input_layer(x.unsqueeze(2))
            x = self.positional_encoding_layer(x)
            x = self.encoder(x)
            return x.flatten(start_dim=1)

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

    def aCpl(self, x):
        """calculate term a in equation Cpl = aCpl + bCpl*T

        Partial molar Cp are from Richet 1985, etc.

        assumes first columns are sio2 al2o3 na2o k2o mgo cao
        """
        # Richet 1985
        # out = (81.37*x[:,0] # Cp liquid SiO2
        #        + 130.2*x[:,1] # Cp liquid Al2O3 (Courtial R. 1993)
        #        + 100.6*x[:,2] # Cp liquid Na2O (Richet 1985)
        #        + 50.13*x[:,3] + x[:,0]*(x[:,3]*x[:,3])*151.7 # Cp liquid K2O (Richet 1985)
        #        + 85.78*x[:,4] # Cp liquid MgO (Richet 1985)
        #        + 86.05*x[:,5] # Cp liquid CaO (Richet 1985)
        #       )

        # solution with a_cp values from neural net
        a_cp = torch.exp(self.out_thermo(self.forward(x))[:, 22:28])
        out = (
            a_cp[:, 0] * x[:, 0]  # Cp liquid SiO2, fixed value from Richet 1984
            + a_cp[:, 1] * x[:, 1]  # Cp liquid Al2O3
            + a_cp[:, 2] * x[:, 2]  # Cp liquid Na2O
            + a_cp[:, 3] * x[:, 3]  # Cp liquid K2O
            + a_cp[:, 4] * x[:, 4]  # Cp liquid MgO
            + a_cp[:, 5] * x[:, 5]  # Cp liquid CaO)
        )

        # solution with a_cp values as global parameters
        # out = (torch.exp(self.Cp_coefs[0])*x[:,0] + # Cp liquid SiO2
        #        torch.exp(self.Cp_coefs[1])*x[:,1] + # Cp liquid Al2O3
        #        torch.exp(self.Cp_coefs[2])*x[:,2] + # Cp liquid Na2O
        #        torch.exp(self.Cp_coefs[3])*x[:,3] + # Cp liquid K2O
        #        torch.exp(self.Cp_coefs[4])*x[:,4] + # Cp liquid MgO
        #        torch.exp(self.Cp_coefs[5])*x[:,5] # Cp liquid CaO)
        #       )

        return torch.reshape(out, (out.shape[0], 1))

    def bCpl(self, x):
        """calculate term b in equation Cpl = aCpl + bCpl*T

        assumes first columns are sio2 al2o3 na2o k2o mgo cao

        only apply B terms on Al and K
        """
        # Richet 1985
        # out = 0.09428*x[:,1] + 0.01578*x[:,3]

        # solution with a_cp values from neural net
        b_cp = torch.exp(self.out_thermo(self.forward(x))[:, 28:30])
        out = b_cp[:, 0] * x[:, 1] + b_cp[:, 1] * x[:, 3]

        # solution with a_cp values as global parameters
        # out = torch.exp(self.Cp_coefs[6])*x[:,1] + torch.exp(self.Cp_coefs[7])*x[:,3]

        return torch.reshape(out, (out.shape[0], 1))

    def cpg_tg(self, x):
        """Glass heat capacity at Tg calculated from Dulong and Petit limit"""
        return 3.0 * 8.314462 * self.at_gfu(x)

    def cpl(self, x, T):
        """Liquid heat capacity at T"""
        out = self.aCpl(x) + self.bCpl(x) * T
        return torch.reshape(out, (out.shape[0], 1))

    def partial_cpl(self, x):
        """partial molar values for Cpl
        6 values in order: SiO2 Al2O3 Na2O K2O MgO CaO
        2 last values are temperature dependence for Al2O3 and K2O
        """
        return torch.exp(self.out_thermo(self.forward(x))[:, 22:30])

    def ap_calc(self, x):
        """calculate term ap in equation dS = ap ln(T/Tg) + b(T-Tg)"""
        out = self.aCpl(x) - self.cpg_tg(x)
        return torch.reshape(out, (out.shape[0], 1))

    def dCp(self, x, T):
        out = self.ap_calc(x) * (torch.log(T) - torch.log(self.tg(x))) + self.bCpl(
            x
        ) * (T - self.tg(x))
        return torch.reshape(out, (out.shape[0], 1))

    def raman_pred(self, x):
        """Raman predicted spectra"""
        return self.out_raman(self.forward(x))

    def tg(self, x):
        """glass transition temperature Tg"""
        out = torch.exp(self.out_thermo(self.forward(x))[:, 0])
        return torch.reshape(out, (out.shape[0], 1))

    def sctg(self, x):
        """configurational entropy at Tg"""
        out = torch.exp(self.out_thermo(self.forward(x))[:, 1])
        return torch.reshape(out, (out.shape[0], 1))

    def ae(self, x):
        """Ae parameter in Adam and Gibbs and MYEGA"""
        out = self.out_thermo(self.forward(x))[:, 2]
        return torch.reshape(out, (out.shape[0], 1))

    def a_am(self, x):
        """A parameter for Avramov-Mitchell"""
        out = self.out_thermo(self.forward(x))[:, 3]
        return torch.reshape(out, (out.shape[0], 1))

    def a_cg(self, x):
        """A parameter for Free Volume (CG)"""
        out = self.out_thermo(self.forward(x))[:, 4]
        return torch.reshape(out, (out.shape[0], 1))

    def a_tvf(self, x):
        """A parameter for VFT"""
        out = self.out_thermo(self.forward(x))[:, 5]
        return torch.reshape(out, (out.shape[0], 1))

    def to_cg(self, x):
        """To parameter for Free Volume (CG)"""
        out = torch.exp(self.out_thermo(self.forward(x))[:, 6])
        return torch.reshape(out, (out.shape[0], 1))

    def c_cg(self, x):
        """C parameter for Free Volume (CG)"""
        out = torch.exp(self.out_thermo(self.forward(x))[:, 7])
        return torch.reshape(out, (out.shape[0], 1))

    def c_tvf(self, x):
        """C parameter for VFT"""
        out = torch.exp(self.out_thermo(self.forward(x))[:, 8])
        return torch.reshape(out, (out.shape[0], 1))

    def vm_glass(self, x):
        """partial molar volume of oxide cations in glass"""
        out = torch.exp(self.out_thermo(self.forward(x))[:, 9:15])
        return torch.reshape(out, (out.shape[0], 6))

    def density_glass(self, x):
        """glass density

        assumes X first columns are sio2 al2o3 na2o k2o mgo cao
        """
        vm_ = self.vm_glass(x)  # partial molar volumes
        w = imelt.molarweights()  # weights

        # calculation of glass molar volume
        v_g = (
            vm_[:, 0] * x[:, 0]
            + vm_[:, 1] * x[:, 1]  # sio2 + al2o3
            + vm_[:, 2] * x[:, 2]
            + vm_[:, 3] * x[:, 3]  # na2o + k2o
            + vm_[:, 4] * x[:, 4]
            + vm_[:, 5] * x[:, 5]
        )  # mgo + cao

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
        out = torch.exp(self.out_thermo(self.forward(x))[:, 15])
        return torch.reshape(out, (out.shape[0], 1))

    def S_B1(self, x):
        """Sellmeir B1"""
        out = self.out_thermo(self.forward(x))[:, 16]
        return torch.reshape(out, (out.shape[0], 1))

    def S_B2(self, x):
        """Sellmeir B2"""
        out = self.out_thermo(self.forward(x))[:, 17]
        return torch.reshape(out, (out.shape[0], 1))

    def S_B3(self, x):
        """Sellmeir B3"""
        out = self.out_thermo(self.forward(x))[:, 18]
        return torch.reshape(out, (out.shape[0], 1))

    def S_C1(self, x):
        """Sellmeir C1, with proper scaling"""
        out = 0.01 * self.out_thermo(self.forward(x))[:, 19]
        return torch.reshape(out, (out.shape[0], 1))

    def S_C2(self, x):
        """Sellmeir C2, with proper scaling"""
        out = 0.1 * self.out_thermo(self.forward(x))[:, 20]

        return torch.reshape(out, (out.shape[0], 1))

    def S_C3(self, x):
        """Sellmeir C3, with proper scaling"""
        out = 100 * self.out_thermo(self.forward(x))[:, 21]
        return torch.reshape(out, (out.shape[0], 1))

    def elastic_modulus(self, x):
        """elastic modulus"""
        out = torch.exp(self.out_thermo(self.forward(x))[:, 30])
        return torch.reshape(out, (out.shape[0], 1))

    def cte(self, x):
        """coefficient of thermal expansion"""
        out = torch.exp(self.out_thermo(self.forward(x))[:, 31])
        return torch.reshape(out, (out.shape[0], 1))

    def abbe(self, x):
        """Abbe number"""
        out = torch.exp(self.out_thermo(self.forward(x))[:, 32])
        return torch.reshape(out, (out.shape[0], 1))

    def liquidus(self, x):
        """liquidus temperature, K"""
        out = torch.exp(self.out_thermo(self.forward(x))[:, 33])
        return torch.reshape(out, (out.shape[0], 1))

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

        need for speed = we decompose the calculation as much as reasonable for a minimum amount of forward pass
        """
        # one forward pass to get thermodynamic output
        thermo_out = self.out_thermo(self.forward(x))

        # get Ae
        ae = torch.reshape(thermo_out[:, 2], (thermo_out[:, 2].shape[0], 1))

        # get ScTg
        sctg = torch.exp(thermo_out[:, 1])
        sctg = torch.reshape(sctg, (sctg.shape[0], 1))

        # get Tg
        tg = torch.exp(thermo_out[:, 0])
        tg = torch.reshape(tg, (tg.shape[0], 1))

        # get Be
        be = (12.0 - ae) * (tg * sctg)

        return ae + be / (T * (sctg + self.dCp(x, T)))

    def myega(self, x, T):
        """viscosity from the MYEGA equation, given entries X and temperature T

        need for speed = we decompose the calculation for making only one forward pass
        """
        # one forward pass to get thermodynamic output
        thermo_out = self.out_thermo(self.forward(x))

        # get Ae
        ae = torch.reshape(thermo_out[:, 2], (thermo_out[:, 2].shape[0], 1))

        # get Tg
        tg = torch.exp(thermo_out[:, 0])
        tg = torch.reshape(tg, (tg.shape[0], 1))

        # get fragility
        frag = torch.exp(thermo_out[:, 15])
        frag = torch.reshape(frag, (frag.shape[0], 1))

        return ae + (12.0 - ae) * (tg / T) * torch.exp(
            (frag / (12.0 - ae) - 1.0) * (tg / T - 1.0)
        )

    def am(self, x, T):
        """viscosity from the Avramov-Mitchell equation, given entries X and temperature T

        need for speed = we decompose the calculation for making only one forward pass
        """

        # one forward pass to get thermodynamic output
        thermo_out = self.out_thermo(self.forward(x))

        # get the A_am parameter
        a_am = torch.reshape(thermo_out[:, 3], (thermo_out[:, 3].shape[0], 1))

        # Get TG
        tg = torch.exp(thermo_out[:, 0])
        tg = torch.reshape(tg, (tg.shape[0], 1))

        # get fragility
        frag = torch.exp(thermo_out[:, 15])
        frag = torch.reshape(frag, (frag.shape[0], 1))

        return a_am + (12.0 - a_am) * (tg / T) ** (frag / (12.0 - a_am))

    def cg(self, x, T):
        """free volume theory viscosity equation, given entries X and temperature T

        need for speed = we decompose the calculation for making only one forward pass
        """

        # one forward pass to get thermodynamic output
        thermo_out = self.out_thermo(self.forward(x))

        # get A CG
        a_cg = torch.reshape(thermo_out[:, 4], (thermo_out[:, 4].shape[0], 1))

        # Get TG
        tg = torch.exp(thermo_out[:, 0])
        tg = torch.reshape(tg, (tg.shape[0], 1))

        # get To
        to_cg = torch.exp(thermo_out[:, 6])
        to_cg = torch.reshape(to_cg, (to_cg.shape[0], 1))

        # get C CG
        c_cg = torch.exp(thermo_out[:, 7])
        c_cg = torch.reshape(c_cg, (c_cg.shape[0], 1))

        b_cg = (
            0.5
            * (12.0 - a_cg)
            * (tg - to_cg + torch.sqrt((tg - to_cg) ** 2 + c_cg * tg))
        )
        return a_cg + 2.0 * b_cg / (T - to_cg + torch.sqrt((T - to_cg) ** 2 + c_cg * T))

    def tvf(self, x, T):
        """Tamman-Vogel-Fulscher empirical viscosity, given entries X and temperature T

        need for speed = we decompose the calculation for making only one forward pass
        """

        # one forward pass to get thermodynamic output
        thermo_out = self.out_thermo(self.forward(x))

        # get the A_tvf parameter
        a_tvf = torch.reshape(thermo_out[:, 5], (thermo_out[:, 5].shape[0], 1))

        # Get TG
        tg = torch.exp(thermo_out[:, 0])
        tg = torch.reshape(tg, (tg.shape[0], 1))

        # Get C_tvf
        c_tvf = torch.exp(thermo_out[:, 8])
        c_tvf = torch.reshape(c_tvf, (c_tvf.shape[0], 1))

        return a_tvf + ((12.0 - a_tvf) * (tg - c_tvf)) / (T - c_tvf)

    def sellmeier(self, x, lbd):
        """Sellmeier equation for refractive index calculation, with lbd in microns"""
        return torch.sqrt(
            1.0
            + self.S_B1(x) * lbd**2 / (lbd**2 - self.S_C1(x))
            + self.S_B2(x) * lbd**2 / (lbd**2 - self.S_C2(x))
            + self.S_B3(x) * lbd**2 / (lbd**2 - self.S_C3(x))
        )


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
            y_ag_pred_train = neuralmodel.ag(x_visco_train, T_visco_train)
            y_myega_pred_train = neuralmodel.myega(x_visco_train, T_visco_train)
            y_am_pred_train = neuralmodel.am(x_visco_train, T_visco_train)
            y_cg_pred_train = neuralmodel.cg(x_visco_train, T_visco_train)
            y_tvf_pred_train = neuralmodel.tvf(x_visco_train, T_visco_train)
            y_raman_pred_train = neuralmodel.raman_pred(x_raman_train)
            y_density_pred_train = neuralmodel.density_glass(x_density_train)
            y_entro_pred_train = neuralmodel.sctg(x_entro_train)
            y_ri_pred_train = neuralmodel.sellmeier(x_ri_train, lbd_ri_train)
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
            y_ag_pred_valid = neuralmodel.ag(
                ds.x_visco_valid.to(device), ds.T_visco_valid.to(device)
            )
            y_myega_pred_valid = neuralmodel.myega(
                ds.x_visco_valid.to(device), ds.T_visco_valid.to(device)
            )
            y_am_pred_valid = neuralmodel.am(
                ds.x_visco_valid.to(device), ds.T_visco_valid.to(device)
            )
            y_cg_pred_valid = neuralmodel.cg(
                ds.x_visco_valid.to(device), ds.T_visco_valid.to(device)
            )
            y_tvf_pred_valid = neuralmodel.tvf(
                ds.x_visco_valid.to(device), ds.T_visco_valid.to(device)
            )
            y_raman_pred_valid = neuralmodel.raman_pred(ds.x_raman_valid.to(device))
            y_density_pred_valid = neuralmodel.density_glass(
                ds.x_density_valid.to(device)
            )
            y_entro_pred_valid = neuralmodel.sctg(ds.x_entro_valid.to(device))
            y_ri_pred_valid = neuralmodel.sellmeier(
                ds.x_ri_valid.to(device), ds.lbd_ri_valid.to(device)
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


def training_lbfgs(
    neuralmodel,
    ds,
    criterion,
    optimizer,
    save_switch=True,
    save_name="./temp",
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

    x_visco_train = ds.x_visco_train.to(device)
    y_visco_train = ds.y_visco_train.to(device)
    T_visco_train = ds.T_visco_train.to(device)

    x_raman_train = ds.x_raman_train.to(device)
    y_raman_train = ds.y_raman_train.to(device)

    x_density_train = ds.x_density_train.to(device)
    y_density_train = ds.y_density_train.to(device)

    x_elastic_train = ds.x_elastic_train.to(device)
    y_elastic_train = ds.y_elastic_train.to(device)

    x_entro_train = ds.x_entro_train.to(device)
    y_entro_train = ds.y_entro_train.to(device)

    x_ri_train = ds.x_ri_train.to(device)
    y_ri_train = ds.y_ri_train.to(device)
    lbd_ri_train = ds.lbd_ri_train.to(device)

    x_cpl_train = ds.x_cpl_train.to(device)
    y_cpl_train = ds.y_cpl_train.to(device)
    T_cpl_train = ds.T_cpl_train.to(device)

    x_cte_train = ds.x_cte_train.to(device)
    y_cte_train = ds.y_cte_train.to(device)

    x_abbe_train = ds.x_abbe_train.to(device)
    y_abbe_train = ds.y_abbe_train.to(device)

    x_liquidus_train = ds.x_liquidus_train.to(device)
    y_liquidus_train = ds.y_liquidus_train.to(device)

    while val_ex <= train_patience:

        #
        # TRAINING
        #
        def closure():  # closure condition for LBFGS

            # Forward pass on training set
            y_ag_pred_train = neuralmodel.ag(x_visco_train, T_visco_train)
            y_myega_pred_train = neuralmodel.myega(x_visco_train, T_visco_train)
            y_am_pred_train = neuralmodel.am(x_visco_train, T_visco_train)
            y_cg_pred_train = neuralmodel.cg(x_visco_train, T_visco_train)
            y_tvf_pred_train = neuralmodel.tvf(x_visco_train, T_visco_train)
            y_raman_pred_train = neuralmodel.raman_pred(x_raman_train)
            y_density_pred_train = neuralmodel.density_glass(x_density_train)
            y_elastic_pred_train = neuralmodel.elastic_modulus(x_elastic_train)
            y_entro_pred_train = neuralmodel.sctg(x_entro_train)
            y_ri_pred_train = neuralmodel.sellmeier(x_ri_train, lbd_ri_train)
            y_cpl_pred_train = neuralmodel.cpl(x_cpl_train, T_cpl_train)
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

            return loss_fold

        # Update weights
        optimizer.step(closure)  # update weights

        # update the running loss
        loss = closure().item()

        # record global loss (mean of the losses of the training folds)
        record_train_loss.append(loss)

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
            y_ag_pred_valid = neuralmodel.ag(
                ds.x_visco_valid.to(device), ds.T_visco_valid.to(device)
            )
            y_myega_pred_valid = neuralmodel.myega(
                ds.x_visco_valid.to(device), ds.T_visco_valid.to(device)
            )
            y_am_pred_valid = neuralmodel.am(
                ds.x_visco_valid.to(device), ds.T_visco_valid.to(device)
            )
            y_cg_pred_valid = neuralmodel.cg(
                ds.x_visco_valid.to(device), ds.T_visco_valid.to(device)
            )
            y_tvf_pred_valid = neuralmodel.tvf(
                ds.x_visco_valid.to(device), ds.T_visco_valid.to(device)
            )
            y_raman_pred_valid = neuralmodel.raman_pred(ds.x_raman_valid.to(device))
            y_density_pred_valid = neuralmodel.density_glass(
                ds.x_density_valid.to(device)
            )
            y_entro_pred_valid = neuralmodel.sctg(ds.x_entro_valid.to(device))
            y_ri_pred_valid = neuralmodel.sellmeier(
                ds.x_ri_valid.to(device), ds.lbd_ri_valid.to(device)
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
                # print('\nTRAIN -- Raman: {:.3f}, d: {:.3f}, S: {:.3f}, RI: {:.3f}, V: {:.3f}, Cp: {:.3f}, Em: {:.3f}, CTE: {:.3f}, Ab: {:.3f}, Tl: {:.3f}'.format(
                # loss_raman, loss_density, loss_entro,  loss_ri, loss_ag, loss_cpl, loss_elastic, loss_cte, loss_abbe, loss_liquidus
                # ))
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
                    "Epoch {} => loss train {:.2f}, valid {:.2f}".format(
                        epoch, loss, loss_v
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
    # if verbose == True:
    #     time2 = time.time()
    #     print("Running time in seconds:", time2-time1)
    #     print("Scaled loss values are:")
    #     print('\nTRAIN -- Raman: {:.3f}, d: {:.3f}, S: {:.3f}, RI: {:.3f}, V: {:.3f}, Cp: {:.3f}, Em: {:.3f}, CTE: {:.3f}, Ab: {:.3f}, Tl: {:.3f}'.format(
    #             loss_raman, loss_density, loss_entro,  loss_ri, loss_ag, loss_cpl, loss_elastic, loss_cte, loss_abbe, loss_liquidus
    #             ))
    #     print('VALID -- Raman: {:.3f}, d: {:.3f}, S: {:.3f}, RI: {:.3f}, V: {:.3f}, Cp: {:.3f}, Em: {:.3f}, CTE: {:.3f}, Ab: {:.3f}, Tl: {:.3f}\n'.format(
    #             loss_raman_v, loss_density_v, loss_entro_v,  loss_ri_v, loss_ag_v, loss_cpl_v, loss_elastic_v, loss_cte_v, loss_abbe_v, loss_liquidus_v
    #             ))

    return neuralmodel, record_train_loss, record_valid_loss


def record_loss_build(path, list_models, ds, shape="rectangle"):
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
            6, nb_neurons, nb_layers, ds.nb_channels_raman, p_drop=p_drop, shape=shape
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
### BAGGING
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
        method,
        X,
        T=np.array([1000.0]),
        lbd=np.array([500.0]),
        sampling=False,
        n_sample=10,
    ):
        """returns predictions from the n models

        Parameters
        ----------
        method : str
            the property to predict. See imelt code for possibilities. Basically it is a string handle that will be converted to an imelt function.
            For instance, for tg, enter 'tg'.
        X : pandas dataframe
            chemical composition for prediction
        T : 1d numpy array or pandas dataframe
            temperatures for predictions, default = np.array([1000.0,])
        lbd : 1d numpy array or pandas dataframe
            lambdas for Sellmeier equation, default = np.array([500.0,])
        sampling : Bool
            if True, dropout is activated and n_sample random samples will be generated per network.
            This allows performing MC Dropout on the ensemble of models.
        """

        #
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
        elif len(lbd) > 1 and len(X) == 1:
            X = torch.tile(X, (len(lbd), 1))
        elif len(X) > 1 and len(T) == 1:
            T = torch.tile(T, (len(X), 1))
            if len(X) > 1 and len(lbd) == 1:
                lbd = torch.tile(lbd, (len(X), 1))

        #
        # sending models to device also.
        # and we activate dropout if necessary for error sampling
        #
        for i in range(self.n_models):
            self.models[i].to(self.device)
            if sampling == True:
                self.models[i].train()

        with torch.no_grad():
            if method == "raman_pred":
                #
                # For Raman spectra generation
                #
                if sampling == True:
                    out = np.zeros(
                        (len(X), 850, self.n_models, n_sample)
                    )  # problem is defined with a X raman shift of 850 values
                    for i in range(self.n_models):
                        for j in range(n_sample):
                            out[:, :, i, j] = (
                                getattr(self.models[i], method)(X)
                                .cpu()
                                .detach()
                                .numpy()
                            )

                    # reshaping for 3D outputs
                    out = out.reshape(
                        (out.shape[0], out.shape[1], out.shape[2] * out.shape[3])
                    )
                else:
                    out = np.zeros(
                        (len(X), 850, self.n_models)
                    )  # problem is defined with a X raman shift of 850 values
                    for i in range(self.n_models):
                        out[:, :, i] = (
                            getattr(self.models[i], method)(X).cpu().detach().numpy()
                        )
            else:
                #
                # Other parameters (latent or real)
                #

                # sampling activated
                if sampling == True:
                    out = np.zeros((len(X), self.n_models, n_sample))
                    if method in frozenset(
                        ("ag", "myega", "am", "cg", "tvf", "density_melt", "cpl", "dCp")
                    ):
                        for i in range(self.n_models):
                            for j in range(n_sample):
                                out[:, i, j] = (
                                    getattr(self.models[i], method)(X, T)
                                    .cpu()
                                    .detach()
                                    .numpy()
                                    .reshape(-1)
                                )
                    elif method == "sellmeier":
                        for i in range(self.n_models):
                            for j in range(n_sample):
                                out[:, i, j] = (
                                    getattr(self.models[i], method)(X, lbd)
                                    .cpu()
                                    .detach()
                                    .numpy()
                                    .reshape(-1)
                                )
                    elif method == "vm_glass":
                        # we must create a new out tensor because we have a multi-output
                        out = np.zeros((len(X), 6, self.n_models, n_sample))
                        for i in range(self.n_models):
                            for j in range(n_sample):
                                out[:, :, i, j] = (
                                    getattr(self.models[i], method)(X)
                                    .cpu()
                                    .detach()
                                    .numpy()
                                )
                    elif method == "partial_cpl":
                        # we must create a new out tensor because we have a multi-output
                        out = np.zeros((len(X), 8, self.n_models, n_sample))
                        for i in range(self.n_models):
                            for j in range(n_sample):
                                out[:, :, i, j] = (
                                    getattr(self.models[i], method)(X)
                                    .cpu()
                                    .detach()
                                    .numpy()
                                )
                    else:
                        for i in range(self.n_models):
                            for j in range(n_sample):
                                out[:, i, j] = (
                                    getattr(self.models[i], method)(X)
                                    .cpu()
                                    .detach()
                                    .numpy()
                                    .reshape(-1)
                                )

                    # reshaping for 2D outputs
                    if method in frozenset(("vm_glass", "partial_cpl")):
                        out = out.reshape(
                            (out.shape[0], out.shape[1], out.shape[2] * out.shape[3])
                        )
                    else:
                        out = out.reshape((out.shape[0], out.shape[1] * out.shape[2]))

                # no sampling
                else:
                    out = np.zeros((len(X), self.n_models))
                    if method in frozenset(
                        ("ag", "myega", "am", "cg", "tvf", "density_melt", "cpl", "dCp")
                    ):
                        for i in range(self.n_models):
                            out[:, i] = (
                                getattr(self.models[i], method)(X, T)
                                .cpu()
                                .detach()
                                .numpy()
                                .reshape(-1)
                            )
                    elif method == "sellmeier":
                        for i in range(self.n_models):
                            out[:, i] = (
                                getattr(self.models[i], method)(X, lbd)
                                .cpu()
                                .detach()
                                .numpy()
                                .reshape(-1)
                            )
                    elif method == "vm_glass":
                        # we must create a new out tensor because we have a multi-output
                        out = np.zeros((len(X), 6, self.n_models))
                        for i in range(self.n_models):
                            out[:, :, i] = (
                                getattr(self.models[i], method)(X)
                                .cpu()
                                .detach()
                                .numpy()
                            )
                    elif method == "partial_cpl":
                        # we must create a new out tensor because we have a multi-output
                        out = np.zeros((len(X), 8, self.n_models))
                        for i in range(self.n_models):
                            out[:, :, i] = (
                                getattr(self.models[i], method)(X)
                                .cpu()
                                .detach()
                                .numpy()
                            )
                    else:
                        for i in range(self.n_models):
                            out[:, i] = (
                                getattr(self.models[i], method)(X)
                                .cpu()
                                .detach()
                                .numpy()
                                .reshape(-1)
                            )

        # Before leaving this function, we make sure we freeze again the dropout
        for i in range(self.n_models):
            self.models[
                i
            ].eval()  # we make sure we freeze dropout if user does not activate sampling

        # returning our sample
        if sampling == False:
            return np.median(out, axis=out.ndim - 1)
        else:
            return out


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
