# (c) Charles Le Losq et co 2022-2024
# see embedded licence file
# imelt V2.1

import os
import torch
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import rampy as rp
import imelt as imelt

from sklearn.metrics import root_mean_squared_error, median_absolute_error

from scipy.signal import savgol_filter

import matplotlib.pyplot as plt


####
# FUNCTIONS FOR DEVICE AND DIR CREATION
####
def get_default_device():
    # First we check if CUDA is available
    print("CUDA AVAILABLE? ")
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        print("Yes, setting device to cuda")
        return torch.device("cuda")
    else:
        print("No, setting device to cpu")
        return torch.device("cpu")


def create_dir(dirName):
    """search and, if necessary, create a folder

    Parameters
    ----------
    dirName : str
        path of the new directory
    """
    try:
        # Create target Directory
        os.mkdir(dirName)
        print("Directory ", dirName, " Created ")
    except FileExistsError:
        print("Directory ", dirName, " already exists")


####
# FUNCTIONS FOR CHEMICAL PREPROCESSING
####
def molarweights():
    """returns a partial table of molecular weights for elements and oxides that can be used in other functions

    Returns
    =======
    w : dictionary
        containing the molar weights of elements and oxides:

        - si, ti, al, fe, li, na, k, mg, ca, ba, o (no upper case, symbol calling)

        - sio2, tio2, al2o3, fe2o3, feo, li2o, na2o, k2o, mgo, cao, sro, bao (no upper case, symbol calling)

    """
    w = {"si": 28.085}

    # From IUPAC Periodic Table 2016, in g/mol
    w["ti"] = 47.867
    w["al"] = 26.982
    w["fe"] = 55.845
    w["h"] = 1.00794
    w["li"] = 6.94
    w["na"] = 22.990
    w["k"] = 39.098
    w["mg"] = 24.305
    w["ca"] = 40.078
    w["ba"] = 137.327
    w["sr"] = 87.62
    w["o"] = 15.9994

    w["ni"] = 58.6934
    w["mn"] = 54.938045
    w["p"] = 30.973762

    # oxides
    w["sio2"] = w["si"] + 2 * w["o"]
    w["tio2"] = w["ti"] + 2 * w["o"]
    w["al2o3"] = 2 * w["al"] + 3 * w["o"]
    w["fe2o3"] = 2 * w["fe"] + 3 * w["o"]
    w["feo"] = w["fe"] + w["o"]
    w["h2o"] = 2 * w["h"] + w["o"]
    w["li2o"] = 2 * w["li"] + w["o"]
    w["na2o"] = 2 * w["na"] + w["o"]
    w["k2o"] = 2 * w["k"] + w["o"]
    w["mgo"] = w["mg"] + w["o"]
    w["cao"] = w["ca"] + w["o"]
    w["sro"] = w["sr"] + w["o"]
    w["bao"] = w["ba"] + w["o"]

    w["nio"] = w["ni"] + w["o"]
    w["mno"] = w["mn"] + w["o"]
    w["p2o5"] = w["p"] * 2 + w["o"] * 5
    return w  # explicit return


def wt_mol(data):
    """to convert weights in mol fraction

    Parameters
    ==========
    data: Pandas DataFrame
            containing the fields sio2,al2o3,na2o,k2o,mgo,cao

    Returns
    =======
    chemtable: Pandas DataFrame
            contains the fields sio2,al2o3,na2o,k2o,mgo,cao in mol%
    """

    chemtable = data.copy()
    w = molarweights()

    # conversion to mol in 100 grammes
    sio2 = chemtable["sio2"] / w["sio2"]
    al2o3 = chemtable["al2o3"] / w["al2o3"]
    na2o = chemtable["na2o"] / w["na2o"]
    k2o = chemtable["k2o"] / w["k2o"]
    mgo = chemtable["mgo"] / w["mgo"]
    cao = chemtable["cao"] / w["cao"]

    # renormalisation
    tot = sio2 + al2o3 + na2o + k2o + mgo + cao
    chemtable["sio2"] = sio2 / tot
    chemtable["al2o3"] = al2o3 / tot
    chemtable["na2o"] = na2o / tot
    chemtable["k2o"] = k2o / tot
    chemtable["mgo"] = mgo / tot
    chemtable["cao"] = cao / tot

    return chemtable


def descriptors(X):
    """generate a X augmented dataframe with new descriptors"""

    T = X.sio2 + 2 * X.al2o3
    O = 2 * X.sio2 + 3 * X.al2o3 + X.na2o + X.k2o + X.mgo + X.cao

    # calculation of NBO/T
    # we allow it to be negative, as it is the case for some samples
    # we will use it as a feature, not as a target
    X["nbot"] = (2 * O - 4 * T) / T

    # calculation of optical basicity
    # partial oxyde values from Moretti 2005,
    # DOI 10.4401/ag-3221
    # Table 1
    X["optbas"] = (
        0.48 * X.sio2
        + 0.59 * X.al2o3
        + 1.15 * X.na2o
        + 1.36 * X.k2o
        + 0.78 * X.mgo
        + 0.99 * X.cao
    )

    # calculation of the ratio of each oxide
    list_oxide = ["sio2", "al2o3", "na2o", "k2o", "mgo", "cao"]
    for i in list_oxide:
        for j in list_oxide:
            if i != j:
                X[i + "_" + j] = X.loc[:, i] / (X.loc[:, i] + X.loc[:, j])

    # calculation of the ratio of aluminium over the sum of metal cations
    X["al_m"] = X.al2o3 / (
        X.al2o3 + X.loc[:, ["na2o", "k2o", "mgo", "cao"]].sum(axis=1)
    )

    return X.fillna(value=0)


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
    list_oxides = ["sio2", "al2o3", "na2o", "k2o", "mgo", "cao"]
    datalist = data.copy()  # safety network

    for i in list_oxides:
        try:
            oxd = datalist[i]
        except:
            datalist[i] = 0.0

    sum_oxides = (
        datalist["sio2"]
        + datalist["al2o3"]
        + datalist["na2o"]
        + datalist["k2o"]
        + datalist["mgo"]
        + datalist["cao"]
    )
    datalist["sio2"] = datalist["sio2"] / sum_oxides
    datalist["al2o3"] = datalist["al2o3"] / sum_oxides
    datalist["na2o"] = datalist["na2o"] / sum_oxides
    datalist["k2o"] = datalist["k2o"] / sum_oxides
    datalist["mgo"] = datalist["mgo"] / sum_oxides
    datalist["cao"] = datalist["cao"] / sum_oxides
    sum_oxides = (
        datalist["sio2"]
        + datalist["al2o3"]
        + datalist["na2o"]
        + datalist["k2o"]
        + datalist["mgo"]
        + datalist["cao"]
    )
    datalist["sum"] = sum_oxides

    return datalist


####
# PREPROCESSING FUNCTIONS
def stratified_group_splitting(
    dataset, target, verbose=False, random_state=167, n_splits=5
):
    """performs a stratified group splitting of the dataset

    Parameters
    ----------
    dataset : pandas dataframe
        dataset to split
    target : str
        name of the target column
    verbose : bool
        if True, prints the number of samples in each set
    random_state : int
        random seed
    n_splits : int
        number of splits to perform
    """

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, random_state=random_state, shuffle=True
    )
    for i, (train_index, vt_index) in enumerate(
        sgkf.split(dataset, class_data(dataset), dataset[target])
    ):
        if i == 0:  # we grab the first fold
            t_i, tv_i = train_index, vt_index

    dataset_train = dataset.loc[t_i, :].reset_index()
    dataset_vt = dataset.loc[tv_i, :].reset_index()

    sgkf = StratifiedGroupKFold(n_splits=2, random_state=random_state, shuffle=True)
    for i, (valid_index, test_index) in enumerate(
        sgkf.split(dataset_vt, class_data(dataset_vt), dataset_vt.Name)
    ):
        if i == 0:  # we grab the first fold
            v_i, ts_i = valid_index, test_index

    dataset_valid = dataset_vt.loc[v_i, :].reset_index()
    dataset_test = dataset_vt.loc[ts_i, :].reset_index()

    if verbose == True:

        nb_train_compo = len(dataset_train["Name"].unique())
        nb_valid_compo = len(dataset_valid["Name"].unique())
        nb_test_compo = len(dataset_test["Name"].unique())
        nb_tot = nb_train_compo + nb_valid_compo + nb_test_compo
        print("Unique compositions in the train, valid and test subsets:")
        print(
            "train {}, valid {}, test {}".format(
                nb_train_compo, nb_valid_compo, nb_test_compo
            )
        )
        print("this makes:")
        print(
            "train {:.1f}%, valid {:.1f}%, test {:.1f}%".format(
                nb_train_compo / nb_tot * 100,
                nb_valid_compo / nb_tot * 100,
                nb_test_compo / nb_tot * 100,
            )
        )

        print("\nDetection of group (composition) leackage: between\n and train-test:")
        print(
            "{} leacked composition between train and valid subsets".format(
                np.sum(dataset_train.Name.isin(dataset_valid.Name).astype(int))
            )
        )
        print(
            "{} leacked composition between train and test subsets".format(
                np.sum(dataset_train.Name.isin(dataset_test.Name).astype(int))
            )
        )

    return dataset_train, dataset_valid, dataset_test


def class_data(chemical_set):
    """class data in different chemical systems

    Parameters
    ----------
    chemical_set : pandas dataframe
        a dataframe containing the chemical data

    Returns
    -------
    class_of_data : numpy array
        an array containing the class of each sample
    """
    # get only the relevant things from chemical_set
    chemical_set = chemical_set.loc[
        :, ["sio2", "al2o3", "na2o", "k2o", "mgo", "cao"]
    ].values

    # an integer array to contain my classes, initialized to 0
    class_of_data = np.zeros(len(chemical_set), dtype=int)

    # sio2-al2o3 (sio2 included), class 1
    class_of_data[
        (chemical_set[:, [0, 1]] >= 0).all(axis=1)
        & (chemical_set[:, [2, 3, 4, 5]] == 0).all(axis=1)
    ] = 1

    # sio2-na2o, class 2
    class_of_data[
        (chemical_set[:, [0, 2]] > 0).all(axis=1)
        & (chemical_set[:, [1, 3, 4, 5]] == 0).all(axis=1)
    ] = 2

    # sio2-k2o, class 3
    class_of_data[
        (chemical_set[:, [0, 3]] > 0).all(axis=1)
        & (chemical_set[:, [1, 2, 4, 5]] == 0).all(axis=1)
    ] = 3

    # sio2-mgo, class 4
    class_of_data[
        (chemical_set[:, [0, 4]] > 0).all(axis=1)
        & (chemical_set[:, [1, 2, 3, 5]] == 0).all(axis=1)
    ] = 4

    # sio2-cao, class 5
    class_of_data[
        (chemical_set[:, [0, 5]] > 0).all(axis=1)
        & (chemical_set[:, [1, 2, 3, 4]] == 0).all(axis=1)
    ] = 5

    # sio2-alkali ternary, class 6
    class_of_data[
        (chemical_set[:, [0, 2, 3]] > 0).all(axis=1)
        & (chemical_set[:, [1, 4, 5]] == 0).all(axis=1)
    ] = 6

    # sio2-alkaline-earth ternary, class 7
    class_of_data[
        (chemical_set[:, [0, 4, 5]] > 0).all(axis=1)
        & (chemical_set[:, [1, 4, 5]] == 0).all(axis=1)
    ] = 7

    # ternaries sio2- MO - M2O, all in class 8
    class_of_data[
        (chemical_set[:, [0, 2, 4]] > 0).all(axis=1)
        & (chemical_set[:, [1, 3, 5]] == 0).all(axis=1)
    ] = 8

    class_of_data[
        (chemical_set[:, [0, 2, 5]] > 0).all(axis=1)
        & (chemical_set[:, [1, 3, 4]] == 0).all(axis=1)
    ] = 8

    class_of_data[
        (chemical_set[:, [0, 3, 4]] > 0).all(axis=1)
        & (chemical_set[:, [1, 2, 5]] == 0).all(axis=1)
    ] = 8

    class_of_data[
        (chemical_set[:, [0, 3, 5]] > 0).all(axis=1)
        & (chemical_set[:, [1, 2, 4]] == 0).all(axis=1)
    ] = 8

    # al2o3-cao, class 9
    class_of_data[
        (chemical_set[:, [1, 5]] > 0).all(axis=1)
        & (chemical_set[:, [0, 2, 3, 4]] == 0).all(axis=1)
    ] = 9

    # sio2-al2o3-na2o, class 10
    class_of_data[
        (chemical_set[:, [0, 1, 2]] > 0).all(axis=1)
        & (chemical_set[:, [3, 4, 5]] == 0).all(axis=1)
    ] = 10

    # sio2-al2o3-k2o, class 11
    class_of_data[
        (chemical_set[:, [0, 1, 3]] > 0).all(axis=1)
        & (chemical_set[:, [2, 4, 5]] == 0).all(axis=1)
    ] = 11

    # sio2-al2o3-na2o-k2o, class 12
    class_of_data[
        (chemical_set[:, [0, 1, 2, 3]] > 0).all(axis=1)
        & (chemical_set[:, [4, 5]] == 0).all(axis=1)
    ] = 12

    # sio2-al2o3-mgo, class 13
    class_of_data[
        (chemical_set[:, [0, 1, 4]] > 0).all(axis=1)
        & (chemical_set[:, [2, 3, 5]] == 0).all(axis=1)
    ] = 13

    # sio2-al2o3-cao, class 14
    class_of_data[
        (chemical_set[:, [0, 1, 5]] > 0).all(axis=1)
        & (chemical_set[:, [2, 3, 4]] == 0).all(axis=1)
    ] = 14

    # sio2-al2o3-mgo-cao, class 15
    class_of_data[
        (chemical_set[:, [0, 1, 4, 5]] > 0).all(axis=1)
        & (chemical_set[:, [2, 3]] == 0).all(axis=1)
    ] = 15

    return class_of_data


def preprocess_raman(
    my_liste,
    path_spectra="./data/raman/",
    generate_figures=False,
    save_path="./figures/datasets/raman/",
):
    """preprocess the raman spectra, resample them and save a figure of the spectra

    Parameters
    ----------
    my_liste : Pandas dataframe
        the user input list.
    path_spectra : string
        the path where to find the spectra.
    save_path : string
        the path where to save the figures.
    """
    nb_exp = my_liste.shape[0]

    x = np.arange(400.0, 1250.0, 1.0)  # our real x axis, for resampling
    spectra_long = np.ones((len(x), nb_exp))

    for i in range(nb_exp):
        file_name, file_extension = os.path.splitext(my_liste.loc[i, "nom"])

        if file_extension == ".txt" or file_extension == ".TXT":
            sp = np.genfromtxt(path_spectra + my_liste.loc[i, "nom"], skip_header=1)
        elif file_extension == ".csv":
            sp = np.genfromtxt(
                path_spectra + my_liste.loc[i, "nom"], skip_header=1, delimiter=","
            )
        else:
            raise ValueError("Unsupported file extension")

        # we sort the array
        sp = sp[sp[:, 0].argsort()]

        # we apply the long correction
        if my_liste.raw[i] == "yes":
            _, y_long, _ = rp.tlcorrection(
                sp[:, 0],
                sp[:, 1] - np.min(sp[sp[:, 0] > 600, 1]) + 1e-16,
                23.0,
                my_liste.loc[i, "laserfreq"],
                normalisation="intensity",
            )
        elif my_liste.raw[i] == "no":
            y_long = sp[:, 1] - np.min(sp[sp[:, 0] > 600, 1])
        else:
            raise ValueError("Check the column raw, should be yes or no.")

        # resampling
        # sav golf filter on long data
        y_smo = savgol_filter(y_long.ravel(), 15, 2)
        # the resampled vector
        y_resampled = np.zeros(len(x))
        # if lower than the maximum wavelength measured, we use values from a savgol filter
        y_resampled[x < np.max(sp[:, 0])] = rp.resample(
            sp[:, 0], y_smo, x[x < np.max(sp[:, 0])], fill_value="extrapolate"
        )
        p_ = np.polyfit(
            sp[sp[:, 0] > (np.max(sp[:, 0]) - 30), 0],
            y_smo[sp[:, 0] > (np.max(sp[:, 0]) - 30)],
            1,
        )
        # for potential values higher, we just do a linear extrapolation
        y_resampled[x > np.max(sp[:, 0])] = np.polyval(p_, x[x > np.max(sp[:, 0])])
        print("Spectra %i, checking array size: %i" % (i, sp.shape[0]))

        # we get the array position of the minima near 800 automatically
        # idx_min = np.where(y_resampled == np.min(y_resampled[(800<= x)&(x<=1000)]))[0]
        # xmin[i] = x[idx_min]

        # updating the BIR
        # bir = np.array([[xmin[i]-5,xmin[i]+5],[1230,1250]])

        # Fitting the background
        # y_bas, bas = rp.baseline(x, y_resampled,bir,"poly",polynomial_order=1)

        # normalisation
        y_bas = (y_resampled - np.min(y_resampled)) / (
            np.max(y_resampled) - np.min(y_resampled)
        )
        # y_bas = y_resampled/np.sum(y_resampled)*200

        # Assigning the spectra in the output array
        spectra_long[:, i] = y_bas.ravel()

        if generate_figures == True:
            # Making a nice figure and saving it
            plt.figure(figsize=(15, 5))
            plt.suptitle(my_liste.loc[i, "product"])
            plt.subplot(1, 3, 1)
            plt.plot(sp[:, 0], sp[:, 1], "k.", ms=1, label="raw")
            plt.legend(loc="best")
            plt.subplot(1, 3, 2)
            plt.plot(sp[:, 0], y_long, "k.", ms=1, label="long")
            plt.plot(x, y_resampled, "r-", linewidth=1, label="resampled")
            plt.legend(loc="best")
            plt.subplot(1, 3, 3)
            plt.plot(x, y_bas, "b-", label="normalised")
            plt.legend(loc="best")
            plt.savefig(save_path + "{}.pdf".format(my_liste.loc[i, "product"]))
            plt.close()

    return spectra_long


###
# ERROR CALCULATIONS
###
def evaluate_accuracy(
    y, ci_lower=0, ci_upper=0, samples=np.array([]), ci_level=0.95, verbose=False
):
    """Evaluate the accuracy of the error bars.

    Provide either the ci_lower and ci_upper values, or an array of samples.

    If you provide an array, the quantiles will be calculated from the array.
    This will be done with taking into account of the ci_level parameter.

    Note : does not work for 3D Raman arrays

    Parameters
    ----------
    y : array
        the observed Y values.
    ci_lower : array
        the lower confidence interval.
    ci_upper : array
        the upper confidence interval.
    samples : array
        the samples from the MC Dropout.
    ci_level : float
        the level of confidence interval. Default is 0.95.
    verbose : bool
        whether to print the accuracy of the error bars. Default is False.

    Returns
    -------
    ic_acc : float
        the accuracy of the error bars.
    percent_above : float
        the percentage of samples above the lower error bar.
    percent_below : float
        the percentage of samples below the upper error bar.
    """
    if samples.shape[0] != 0:
        ci_lower = np.quantile(samples, (1 - ci_level) / 2, axis=samples.ndim - 1)
        ci_upper = np.quantile(
            samples, ci_level + (1 - ci_level) / 2, axis=samples.ndim - 1
        )
    else:
        if (ci_upper.all() == 0) or (ci_lower.all() == 0):
            raise ValueError("Provide the ci values")

    ic_acc = (ci_lower <= y.ravel()) * (ci_upper >= y.ravel())
    ic_acc = ic_acc.mean()
    percent_above = (ci_lower <= y.ravel()).mean()
    percent_below = (ci_upper >= y.ravel()).mean()

    if verbose == True:
        print("accuracy of MC Dropout error bars: {.3f}".format(ic_acc))
        print(
            "Percentage of samples above lower c.i.:".format(
                (ci_lower <= y.ravel()).float().mean()
            )
        )
        print(
            "Percentage of samples below upper c.i.:".format(
                (ci_upper >= y.ravel()).float().mean()
            )
        )
    return ic_acc, percent_above, percent_below


def residual_error_calc(y, y_pred, mode="BOTH"):
    """returns the root-mean-squared-error (RMSE) and median absolute error (MAE) between y an y_pred

    Parameters
    ----------
    y : array-like
        the true values
    y_pred : array-like
        the predicted values
    mode : str, optional
        the type of error to be calculated (RMSE, MAE or BOTH). By default "BOTH"

    Returns
    -------
    float or tuple
        the error or a tuple of errors

    """

    # check the type of input arrays and convert them to numpy if necessary
    if type(y) == torch.Tensor:
        y = y.cpu().detach().numpy()
    if type(y_pred) == torch.Tensor:
        y_pred = y_pred.cpu().detach().numpy()

    # now perform the relevant calculation
    if mode == "RMSE":  # root mean square error
        return root_mean_squared_error(y, y_pred)
    elif mode == "MAE":  # median absolute deviation
        return median_absolute_error(y, y_pred)
    elif mode == "BOTH":
        rmse = root_mean_squared_error(y, y_pred)
        mae = median_absolute_error(y, y_pred)
        return rmse, mae


def error_viscosity_bydomain(bagged_model, ds, methods=["ag", "tvf"], boundary=7.0, mode="RMSE"):
    """return the root mean squared error or the median absolute error between predicted and measured viscosities

    Parameters
    ----------
    bagged_model : bagged model object
        generated using the bagging_models class
    ds : ds dataset
        contains all the training, validation and test viscosity datasets
    methods : list
        list of viscosity calculation methods to provide to the bagged model
        choose between "ag", "tvf", "am", "myega", "cg"
    boundary : float
        boundary between the high and low viscosity domains (log Pa s value)
    mode : str
        method to use for the error calculation, either "RMSE" for root_mean_squared_error or "MAE" for median_absolute_error
    """
    preds_train = bagged_model.predict(
        methods, ds.x_visco_train, ds.T_visco_train
    )
    preds_valid = bagged_model.predict(
        methods, ds.x_visco_valid, ds.T_visco_valid
    )
    preds_test = bagged_model.predict(
        methods, ds.x_visco_test, ds.T_visco_test
    )

    # get viscosity data
    y_train = ds.y_visco_train
    y_valid = ds.y_visco_valid
    y_test = ds.y_visco_test


    # now compare them to predictions
    for method in methods:
        y_pred_train = preds_train[method].reshape(-1,1)
        y_pred_valid = preds_valid[method].reshape(-1,1)
        y_pred_test = preds_test[method].reshape(-1,1)

        total_RMSE_train = residual_error_calc(y_pred_train, y_train, mode=mode)
        total_RMSE_valid = residual_error_calc(y_pred_valid, y_valid, mode=mode)
        total_RMSE_test = residual_error_calc(y_pred_test, y_test, mode=mode)

        high_RMSE_train = residual_error_calc(
            y_pred_train[y_train > boundary], y_train[y_train > boundary], mode=mode
        )
        high_RMSE_valid = residual_error_calc(
            y_pred_valid[y_valid > boundary], y_valid[y_valid > boundary], mode=mode
        )
        high_RMSE_test = residual_error_calc(
            y_pred_test[y_test > boundary], y_test[y_test > boundary], mode=mode
        )

        low_RMSE_train = residual_error_calc(
            y_pred_train[y_train < boundary], y_train[y_train < boundary], mode=mode
        )
        low_RMSE_valid = residual_error_calc(
            y_pred_valid[y_valid < boundary], y_valid[y_valid < boundary], mode=mode
        )
        low_RMSE_test = residual_error_calc(
            y_pred_test[y_test < boundary], y_test[y_test < boundary], mode=mode
        )

        if method == "ag":
            name_method = "Adam-Gibbs"
        elif method == "cg":
            name_method = "Free Volume"
        elif method == "tvf":
            name_method = "Vogel Fulcher Tamman"
        elif method == "myega":
            name_method = "MYEGA"
        elif method == "am":
            name_method = "Avramov Milchev"

        print("{} using the equation from {}:".format(mode, name_method))
        print(
            "     on the full range (0-15 log Pa s): train {0:.2f}, valid {1:.2f}, test {2:.2f}".format(
                total_RMSE_train, total_RMSE_valid, total_RMSE_test
            )
        )
        print(
            "     on the -inf - {:.1f} log Pa s range: train {:.2f}, valid {:.2f}, test {:.2f}".format(
                boundary, low_RMSE_train, low_RMSE_valid, low_RMSE_test
            )
        )
        print(
            "     on the {:.1f} - +inf log Pa s range: train {:.2f}, valid {:.2f}, test {:.2f}".format(
                boundary, high_RMSE_train, high_RMSE_valid, high_RMSE_test
            )
        )
        print("")


###
# FUNCTION FOR RAMAN CALCS
###
def R_Raman(x, y, lb=670, hb=870):
    """calculates the R_Raman parameter of a Raman signal y sampled at x.

    y can be an NxM array with N samples and M Raman shifts.
    """
    A_LW = np.trapz(y[:, x < lb], x[x < lb], axis=1)
    A_HW = np.trapz(y[:, x > hb], x[x > hb], axis=1)
    return A_LW / A_HW
