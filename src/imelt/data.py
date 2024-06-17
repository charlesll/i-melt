# (c) Charles Le Losq et co 2022-2024
# see embedded licence file
# imelt V2.1

import numpy as np
import torch
import h5py
import pandas as pd

# to load data in a library
from pathlib import Path
import os

_BASEDATAPATH = Path(os.path.dirname(__file__)) / "data"

###
### FUNCTIONS FOR HANDLNG DATA
###


def list_oxides():
    """return the list of oxide components in the good order"""
    return ["sio2", "al2o3", "na2o", "k2o", "mgo", "cao"]


class data_loader:
    """custom data loader for batch training"""

    def __init__(
        self,
        path_viscosity=_BASEDATAPATH / "NKCMAS_viscosity.hdf5",
        path_raman=_BASEDATAPATH / "NKCMAS_Raman.hdf5",
        path_density=_BASEDATAPATH / "NKCMAS_density.hdf5",
        path_ri=_BASEDATAPATH / "NKCMAS_optical.hdf5",
        path_cp=_BASEDATAPATH / "NKCMAS_cp.hdf5",
        path_elastic=_BASEDATAPATH / "NKCMAS_em.hdf5",
        path_cte=_BASEDATAPATH / "NKCMAS_cte.hdf5",
        path_abbe=_BASEDATAPATH / "NKCMAS_abbe.hdf5",
        path_liquidus=_BASEDATAPATH / "NKCMAS_tl.hdf5",
        scaling=False,
    ):
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
        path_cp : String
            path for the liquid heat capacity HDF5 dataset
        path_elastic : String
            path for the elastic moduli HDF5 dataset
        path_cte : String
            path for the thermal expansion HDF5 dataset
        path_abbe : String
            path for the Abbe number HDF5 dataset
        path_liquidus : String
            path for the liquidus temperature HDF5 dataset
        scaling : False or True
            Scales the input chemical composition.
            WARNING : Does not work currently as this is a relic of testing this effect,
            but we chose not to scale inputs and Cp are calculated with unscaled values in the network.
        """

        f = h5py.File(path_viscosity, "r")

        # List all groups
        self.X_columns = f["X_columns"][()]

        # Entropy dataset
        X_entropy_train = f["X_entropy_train"][()]
        y_entropy_train = f["y_entropy_train"][()]

        X_entropy_valid = f["X_entropy_valid"][()]
        y_entropy_valid = f["y_entropy_valid"][()]

        X_entropy_test = f["X_entropy_test"][()]
        y_entropy_test = f["y_entropy_test"][()]

        # Viscosity dataset
        X_train = f["X_train"][()]
        T_train = f["T_train"][()]
        y_train = f["y_train"][()]

        X_valid = f["X_valid"][()]
        T_valid = f["T_valid"][()]
        y_valid = f["y_valid"][()]

        X_test = f["X_test"][()]
        T_test = f["T_test"][()]
        y_test = f["y_test"][()]

        # Tg dataset
        X_tg_train = f["X_tg_train"][()]
        X_tg_valid = f["X_tg_valid"][()]
        X_tg_test = f["X_tg_test"][()]

        y_tg_train = f["y_tg_train"][()]
        y_tg_valid = f["y_tg_valid"][()]
        y_tg_test = f["y_tg_test"][()]

        f.close()

        # Raman dataset
        f = h5py.File(path_raman, "r")
        X_raman_train = f["X_raman_train"][()]
        y_raman_train = f["y_raman_train"][()]
        X_raman_valid = f["X_raman_valid"][()]
        y_raman_valid = f["y_raman_valid"][()]
        X_raman_test = f["X_raman_test"][()]
        y_raman_test = f["y_raman_test"][()]
        f.close()

        # Raman axis is
        self.x_raman_shift = np.arange(400.0, 1250.0, 1.0)

        # grabbing number of Raman channels
        self.nb_channels_raman = y_raman_valid.shape[1]

        # Density dataset
        f = h5py.File(path_density, "r")
        X_density_train = f["X_density_train"][()]
        X_density_valid = f["X_density_valid"][()]
        X_density_test = f["X_density_test"][()]

        y_density_train = f["y_density_train"][()]
        y_density_valid = f["y_density_valid"][()]
        y_density_test = f["y_density_test"][()]
        f.close()

        # Elastic Modulus dataset
        f = h5py.File(path_elastic, "r")
        X_elastic_train = f["X_elastic_train"][()]
        X_elastic_valid = f["X_elastic_valid"][()]
        X_elastic_test = f["X_elastic_test"][()]

        y_elastic_train = f["y_elastic_train"][()]
        y_elastic_valid = f["y_elastic_valid"][()]
        y_elastic_test = f["y_elastic_test"][()]
        f.close()

        # Thermal expansion dataset
        f = h5py.File(path_cte, "r")
        X_cte_train = f["X_cte_train"][()]
        X_cte_valid = f["X_cte_valid"][()]
        X_cte_test = f["X_cte_test"][()]

        y_cte_train = f["y_cte_train"][()]
        y_cte_valid = f["y_cte_valid"][()]
        y_cte_test = f["y_cte_test"][()]
        f.close()

        # Abbe number dataset
        f = h5py.File(path_abbe, "r")
        X_abbe_train = f["X_abbe_train"][()]
        X_abbe_valid = f["X_abbe_valid"][()]
        X_abbe_test = f["X_abbe_test"][()]

        y_abbe_train = f["y_abbe_train"][()]
        y_abbe_valid = f["y_abbe_valid"][()]
        y_abbe_test = f["y_abbe_test"][()]
        f.close()

        # Refractive Index (ri) dataset
        f = h5py.File(path_ri, "r")
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

        # Liquid heat capacity (cp) dataset
        f = h5py.File(path_cp, "r")
        X_cpl_train = f["X_cpl_train"][()]
        T_cpl_train = f["T_cpl_train"][()]
        y_cpl_train = f["y_cpl_train"][()]

        X_cpl_valid = f["X_cpl_valid"][()]
        T_cpl_valid = f["T_cpl_valid"][()]
        y_cpl_valid = f["y_cpl_valid"][()]

        X_cpl_test = f["X_cpl_test"][()]
        T_cpl_test = f["T_cpl_test"][()]
        y_cpl_test = f["y_cpl_test"][()]
        f.close()

        # Liquidus dataset
        f = h5py.File(path_liquidus, "r")
        X_liquidus_train = f["X_liquidus_train"][()]
        X_liquidus_valid = f["X_liquidus_valid"][()]
        X_liquidus_test = f["X_liquidus_test"][()]

        y_liquidus_train = f["y_liquidus_train"][()]
        y_liquidus_valid = f["y_liquidus_valid"][()]
        y_liquidus_test = f["y_liquidus_test"][()]
        f.close()

        # preparing data for pytorch

        # Scaler
        # Warning : this was done for tests and currently will not work,
        # as Cp are calculated from unscaled mole fractions...
        if scaling == True:
            X_scaler_mean = np.mean(X_train, axis=0)
            X_scaler_std = np.std(X_train, axis=0)
        else:
            X_scaler_mean = 0.0
            X_scaler_std = 1.0

        # The following lines perform scaling (not needed, not active),
        # put the data in torch tensors and send them to device (GPU or CPU, as requested) not anymore

        # viscosity
        self.x_visco_train = torch.FloatTensor(
            self.scaling(X_train, X_scaler_mean, X_scaler_std)
        )
        self.T_visco_train = torch.FloatTensor(T_train.reshape(-1, 1))
        self.y_visco_train = torch.FloatTensor(y_train[:, 0].reshape(-1, 1))

        self.x_visco_valid = torch.FloatTensor(
            self.scaling(X_valid, X_scaler_mean, X_scaler_std)
        )
        self.T_visco_valid = torch.FloatTensor(T_valid.reshape(-1, 1))
        self.y_visco_valid = torch.FloatTensor(y_valid[:, 0].reshape(-1, 1))

        self.x_visco_test = torch.FloatTensor(
            self.scaling(X_test, X_scaler_mean, X_scaler_std)
        )
        self.T_visco_test = torch.FloatTensor(T_test.reshape(-1, 1))
        self.y_visco_test = torch.FloatTensor(y_test[:, 0].reshape(-1, 1))

        # entropy
        self.x_entro_train = torch.FloatTensor(
            self.scaling(X_entropy_train, X_scaler_mean, X_scaler_std)
        )
        self.y_entro_train = torch.FloatTensor(y_entropy_train[:, 0].reshape(-1, 1))

        self.x_entro_valid = torch.FloatTensor(
            self.scaling(X_entropy_valid, X_scaler_mean, X_scaler_std)
        )
        self.y_entro_valid = torch.FloatTensor(y_entropy_valid[:, 0].reshape(-1, 1))

        self.x_entro_test = torch.FloatTensor(
            self.scaling(X_entropy_test, X_scaler_mean, X_scaler_std)
        )
        self.y_entro_test = torch.FloatTensor(y_entropy_test[:, 0].reshape(-1, 1))

        # tg
        self.x_tg_train = torch.FloatTensor(
            self.scaling(X_tg_train, X_scaler_mean, X_scaler_std)
        )
        self.y_tg_train = torch.FloatTensor(y_tg_train.reshape(-1, 1))

        self.x_tg_valid = torch.FloatTensor(
            self.scaling(X_tg_valid, X_scaler_mean, X_scaler_std)
        )
        self.y_tg_valid = torch.FloatTensor(y_tg_valid.reshape(-1, 1))

        self.x_tg_test = torch.FloatTensor(
            self.scaling(X_tg_test, X_scaler_mean, X_scaler_std)
        )
        self.y_tg_test = torch.FloatTensor(y_tg_test.reshape(-1, 1))

        # Glass density
        self.x_density_train = torch.FloatTensor(
            self.scaling(X_density_train, X_scaler_mean, X_scaler_std)
        )
        self.y_density_train = torch.FloatTensor(y_density_train.reshape(-1, 1))

        self.x_density_valid = torch.FloatTensor(
            self.scaling(X_density_valid, X_scaler_mean, X_scaler_std)
        )
        self.y_density_valid = torch.FloatTensor(y_density_valid.reshape(-1, 1))

        self.x_density_test = torch.FloatTensor(
            self.scaling(X_density_test, X_scaler_mean, X_scaler_std)
        )
        self.y_density_test = torch.FloatTensor(y_density_test.reshape(-1, 1))

        # Glass elastic modulus
        self.x_elastic_train = torch.FloatTensor(
            self.scaling(X_elastic_train, X_scaler_mean, X_scaler_std)
        )
        self.y_elastic_train = torch.FloatTensor(y_elastic_train.reshape(-1, 1))

        self.x_elastic_valid = torch.FloatTensor(
            self.scaling(X_elastic_valid, X_scaler_mean, X_scaler_std)
        )
        self.y_elastic_valid = torch.FloatTensor(y_elastic_valid.reshape(-1, 1))

        self.x_elastic_test = torch.FloatTensor(
            self.scaling(X_elastic_test, X_scaler_mean, X_scaler_std)
        )
        self.y_elastic_test = torch.FloatTensor(y_elastic_test.reshape(-1, 1))

        # Glass thermal expansion
        self.x_cte_train = torch.FloatTensor(
            self.scaling(X_cte_train, X_scaler_mean, X_scaler_std)
        )
        self.y_cte_train = torch.FloatTensor(y_cte_train.reshape(-1, 1))

        self.x_cte_valid = torch.FloatTensor(
            self.scaling(X_cte_valid, X_scaler_mean, X_scaler_std)
        )
        self.y_cte_valid = torch.FloatTensor(y_cte_valid.reshape(-1, 1))

        self.x_cte_test = torch.FloatTensor(
            self.scaling(X_cte_test, X_scaler_mean, X_scaler_std)
        )
        self.y_cte_test = torch.FloatTensor(y_cte_test.reshape(-1, 1))

        # Glass Abbe Number
        self.x_abbe_train = torch.FloatTensor(
            self.scaling(X_abbe_train, X_scaler_mean, X_scaler_std)
        )
        self.y_abbe_train = torch.FloatTensor(y_abbe_train.reshape(-1, 1))

        self.x_abbe_valid = torch.FloatTensor(
            self.scaling(X_abbe_valid, X_scaler_mean, X_scaler_std)
        )
        self.y_abbe_valid = torch.FloatTensor(y_abbe_valid.reshape(-1, 1))

        self.x_abbe_test = torch.FloatTensor(
            self.scaling(X_abbe_test, X_scaler_mean, X_scaler_std)
        )
        self.y_abbe_test = torch.FloatTensor(y_abbe_test.reshape(-1, 1))

        # Optical
        self.x_ri_train = torch.FloatTensor(
            self.scaling(X_ri_train, X_scaler_mean, X_scaler_std)
        )
        self.lbd_ri_train = torch.FloatTensor(lbd_ri_train.reshape(-1, 1))
        self.y_ri_train = torch.FloatTensor(y_ri_train.reshape(-1, 1))

        self.x_ri_valid = torch.FloatTensor(
            self.scaling(X_ri_valid, X_scaler_mean, X_scaler_std)
        )
        self.lbd_ri_valid = torch.FloatTensor(lbd_ri_valid.reshape(-1, 1))
        self.y_ri_valid = torch.FloatTensor(y_ri_valid.reshape(-1, 1))

        self.x_ri_test = torch.FloatTensor(
            self.scaling(X_ri_test, X_scaler_mean, X_scaler_std)
        )
        self.lbd_ri_test = torch.FloatTensor(lbd_ri_test.reshape(-1, 1))
        self.y_ri_test = torch.FloatTensor(y_ri_test.reshape(-1, 1))

        # Raman
        self.x_raman_train = torch.FloatTensor(
            self.scaling(X_raman_train, X_scaler_mean, X_scaler_std)
        )
        self.y_raman_train = torch.FloatTensor(y_raman_train)

        self.x_raman_valid = torch.FloatTensor(
            self.scaling(X_raman_valid, X_scaler_mean, X_scaler_std)
        )
        self.y_raman_valid = torch.FloatTensor(y_raman_valid)

        self.x_raman_test = torch.FloatTensor(
            self.scaling(X_raman_test, X_scaler_mean, X_scaler_std)
        )
        self.y_raman_test = torch.FloatTensor(y_raman_test)

        # Liquid heat capacity
        self.x_cpl_train = torch.FloatTensor(
            self.scaling(X_cpl_train, X_scaler_mean, X_scaler_std)
        )
        self.T_cpl_train = torch.FloatTensor(T_cpl_train.reshape(-1, 1))
        self.y_cpl_train = torch.FloatTensor(y_cpl_train.reshape(-1, 1))

        self.x_cpl_valid = torch.FloatTensor(
            self.scaling(X_cpl_valid, X_scaler_mean, X_scaler_std)
        )
        self.T_cpl_valid = torch.FloatTensor(T_cpl_valid.reshape(-1, 1))
        self.y_cpl_valid = torch.FloatTensor(y_cpl_valid.reshape(-1, 1))

        self.x_cpl_test = torch.FloatTensor(
            self.scaling(X_cpl_test, X_scaler_mean, X_scaler_std)
        )
        self.T_cpl_test = torch.FloatTensor(T_cpl_test.reshape(-1, 1))
        self.y_cpl_test = torch.FloatTensor(y_cpl_test.reshape(-1, 1))

        # Liquidus temperature
        self.x_liquidus_train = torch.FloatTensor(
            self.scaling(X_liquidus_train, X_scaler_mean, X_scaler_std)
        )
        self.y_liquidus_train = torch.FloatTensor(y_liquidus_train.reshape(-1, 1))

        self.x_liquidus_valid = torch.FloatTensor(
            self.scaling(X_liquidus_valid, X_scaler_mean, X_scaler_std)
        )
        self.y_liquidus_valid = torch.FloatTensor(y_liquidus_valid.reshape(-1, 1))

        self.x_liquidus_test = torch.FloatTensor(
            self.scaling(X_liquidus_test, X_scaler_mean, X_scaler_std)
        )
        self.y_liquidus_test = torch.FloatTensor(y_liquidus_test.reshape(-1, 1))

    def recall_order(self):
        print(
            "Order of chemical components is sio2, al2o3, na2o, k2o, mgo, cao, then descriptors"
        )

    def scaling(self, X, mu, s):
        return (X - mu) / s

    def print_data(self):
        """print the specifications of the datasets"""

        print("################################")
        print("#### Dataset specifications ####")
        print("################################")

        # print splitting
        size_train = self.x_visco_train.unique(dim=0).shape[0]
        size_valid = self.x_visco_valid.unique(dim=0).shape[0]
        size_test = self.x_visco_test.unique(dim=0).shape[0]
        size_total = size_train + size_valid + size_test
        self.size_total_visco = size_total

        print("")
        print("Number of unique compositions (viscosity): {}".format(size_total))
        print(
            "Number of unique compositions in training (viscosity): {}".format(
                size_train
            )
        )
        print(
            "Dataset separations are {:.2f} in train, {:.2f} in valid, {:.2f} in test".format(
                size_train / size_total, size_valid / size_total, size_test / size_total
            )
        )

        # print splitting
        size_train = self.x_entro_train.unique(dim=0).shape[0]
        size_valid = self.x_entro_valid.unique(dim=0).shape[0]
        size_test = self.x_entro_test.unique(dim=0).shape[0]
        size_total = size_train + size_valid + size_test
        self.size_total_entro = size_total

        print("")
        print("Number of unique compositions (entropy): {}".format(size_total))
        print(
            "Number of unique compositions in training (entropy): {}".format(size_train)
        )
        print(
            "Dataset separations are {:.2f} in train, {:.2f} in valid, {:.2f} in test".format(
                size_train / size_total, size_valid / size_total, size_test / size_total
            )
        )

        size_train = self.x_ri_train.unique(dim=0).shape[0]
        size_valid = self.x_ri_valid.unique(dim=0).shape[0]
        size_test = self.x_ri_test.unique(dim=0).shape[0]
        size_total = size_train + size_valid + size_test
        self.size_total_ri = size_total

        print("")
        print("Number of unique compositions (refractive index): {}".format(size_total))
        print(
            "Number of unique compositions in training (refractive index): {}".format(
                size_train
            )
        )
        print(
            "Dataset separations are {:.2f} in train, {:.2f} in valid, {:.2f} in test".format(
                size_train / size_total, size_valid / size_total, size_test / size_total
            )
        )

        # Density
        size_train = self.x_density_train.unique(dim=0).shape[0]
        size_valid = self.x_density_valid.unique(dim=0).shape[0]
        size_test = self.x_density_test.unique(dim=0).shape[0]
        size_total = size_train + size_valid + size_test
        self.size_total_density = size_total

        print("")
        print("Number of unique compositions (glass density): {}".format(size_total))
        print(
            "Number of unique compositions in training (glass density): {}".format(
                size_train
            )
        )
        print(
            "Dataset separations are {:.2f} in train, {:.2f} in valid, {:.2f} in test".format(
                size_train / size_total, size_valid / size_total, size_test / size_total
            )
        )

        # Elastic Modulus
        size_train = self.x_elastic_train.unique(dim=0).shape[0]
        size_valid = self.x_elastic_valid.unique(dim=0).shape[0]
        size_test = self.x_elastic_test.unique(dim=0).shape[0]
        size_total = size_train + size_valid + size_test
        self.size_total_elastic = size_total

        print("")
        print(
            "Number of unique compositions (glass elastic modulus): {}".format(
                size_total
            )
        )
        print(
            "Number of unique compositions in training (glass elastic modulus): {}".format(
                size_train
            )
        )
        print(
            "Dataset separations are {:.2f} in train, {:.2f} in valid, {:.2f} in test".format(
                size_train / size_total, size_valid / size_total, size_test / size_total
            )
        )

        # same thing for CTE
        size_train = self.x_cte_train.unique(dim=0).shape[0]
        size_valid = self.x_cte_valid.unique(dim=0).shape[0]
        size_test = self.x_cte_test.unique(dim=0).shape[0]
        size_total = size_train + size_valid + size_test
        self.size_total_cte = size_total

        print("")
        print("Number of unique compositions (glass CTE): {}".format(size_total))
        print(
            "Number of unique compositions in training (glass CTE): {}".format(
                size_train
            )
        )
        print(
            "Dataset separations are {:.2f} in train, {:.2f} in valid, {:.2f} in test".format(
                size_train / size_total, size_valid / size_total, size_test / size_total
            )
        )

        # Abbe Number
        size_train = self.x_abbe_train.unique(dim=0).shape[0]
        size_valid = self.x_abbe_valid.unique(dim=0).shape[0]
        size_test = self.x_abbe_test.unique(dim=0).shape[0]
        size_total = size_train + size_valid + size_test
        self.size_total_abbe = size_total

        print("")
        print(
            "Number of unique compositions (glass Abbe Number): {}".format(size_total)
        )
        print(
            "Number of unique compositions in training (glass Abbe Number): {}".format(
                size_train
            )
        )
        print(
            "Dataset separations are {:.2f} in train, {:.2f} in valid, {:.2f} in test".format(
                size_train / size_total, size_valid / size_total, size_test / size_total
            )
        )

        # LIQUIDUS TEMPERATURE
        size_train = self.x_liquidus_train.unique(dim=0).shape[0]
        size_valid = self.x_liquidus_valid.unique(dim=0).shape[0]
        size_test = self.x_liquidus_test.unique(dim=0).shape[0]
        size_total = size_train + size_valid + size_test
        self.size_total_liquidus = size_total

        print("")
        print(
            "Number of unique compositions (Liquidus temperature): {}".format(
                size_total
            )
        )
        print(
            "Number of unique compositions in training (Liquidus temperature): {}".format(
                size_train
            )
        )
        print(
            "Dataset separations are {:.2f} in train, {:.2f} in valid, {:.2f} in test".format(
                size_train / size_total, size_valid / size_total, size_test / size_total
            )
        )

        # Liquid heat capacity
        size_train = self.x_cpl_train.unique(dim=0).shape[0]
        size_valid = self.x_cpl_valid.unique(dim=0).shape[0]
        size_test = self.x_cpl_test.unique(dim=0).shape[0]
        size_total = size_train + size_valid + size_test
        self.size_total_cpl = size_total

        print("")
        print("Number of unique compositions (heat capacity): {}".format(size_total))
        print(
            "Number of unique compositions in training (heat capacity): {}".format(
                size_train
            )
        )
        # print("Dataset separations are {:.2f} in train, {:.2f} in valid, {:.2f} in test".format(size_train/size_total,
        #                                                                             size_valid/size_total,
        #                                                                             size_test/size_total))

        # RAMAN
        size_train = self.x_raman_train.unique(dim=0).shape[0]
        size_valid = self.x_raman_valid.unique(dim=0).shape[0]
        size_test = self.x_raman_test.unique(dim=0).shape[0]
        size_total = size_train + size_valid + size_test
        self.size_total_raman = size_total

        print("")
        print("Number of unique compositions (Raman): {}".format(size_total))
        print(
            "Number of unique compositions in training (Raman): {}".format(size_train)
        )
        print(
            "Dataset separations are {:.2f} in train, {:.2f} in valid".format(
                size_train / size_total, size_valid / size_total
            )
        )

        # training shapes
        print("")
        print("This is for checking the shape consistency of the dataset:\n")

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

        print("Elastic Modulus train shape")
        print(self.x_elastic_train.shape)
        print(self.y_elastic_train.shape)

        print("CTE train shape")
        print(self.x_cte_train.shape)
        print(self.y_cte_train.shape)

        print("Abbe Number train shape")
        print(self.x_abbe_train.shape)
        print(self.y_abbe_train.shape)

        print("Liquidus temperature train shape")
        print(self.x_liquidus_train.shape)
        print(self.y_liquidus_train.shape)

        print("Refractive Index train shape")
        print(self.x_ri_train.shape)
        print(self.lbd_ri_train.shape)
        print(self.y_ri_train.shape)

        # print shape of cpl
        print("Liquid heat capacity train shape")
        print(self.x_cpl_train.shape)
        print(self.T_cpl_train.shape)
        print(self.y_cpl_train.shape)

        print("Raman train shape")
        print(self.x_raman_train.shape)
        print(self.y_raman_train.shape)
