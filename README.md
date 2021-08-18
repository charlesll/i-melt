# i-Melt

(c) 2021 Charles Le Losq, lelosq@ipgp.fr

## LICENSE

Any material in this repository is under the MIT licence

See MIT license file for details

## REQUIREMENTS

see requirements.txt file

The model and useful functions are contained in the imelt.py file.

### Installation of requirements

Starting from a barebone Python 3 environment with pip installed, simply open a terminal pointing to the working folder and type

`$ pip install --user -r requirements.txt`

## Data

All data are given in a Database_IPGP.xlsx file in the data folder. Raman spectra are contained in a ./data/raman/ folder.

## Data preparation

The notebook **Dataset_preparation.ipynb** allows preparation of the datasets, which are subsequently saved in HDF5 format in the data folder.

The **Dataset_visualization.ipynb** notebook shows the distribution of data in ternary plots.

## Training the networks

### Hyperparameter tuning

- A Random search experiment as well as the experiment about the dataset size are done in the **Experiment_1_architecture.py** and **Experiment_2_dataset_size.py** codes. Due to the large amount of calculations, training is best done on GPU. *Training takes ~72 hours or more on a Dell Precision 5251 equipped with a RTX 4000 NVIDIA GPU.*

- The **Training_BO.ipynb** notebook allows to perform Bayesian Optimization for hyperparameter selection using AX plateform.

### Training candidates

The code **Training_Candidates.py** allows training 100 networks with the reference architecture and selects the 10 best ones, which are saved in ./model/best/ and used for future predictions.

**The 10 best models at the time of publication are already provided in ./model/best/. Please beware that running this notebook will update them and can affect slightly the results because each neural network training begins from a different, randomly generated starting point.**

## Repeating the result analysis of Le Losq et al. GCA 2021

Analysis of results and predictions from the 10 best trained networks is done in several steps:

- The **Results_experiments.ipynb** code allows observing the results of the random search and dataset size experiments in Supplementary Figure 1 (see ./figures folder). **Please note that the originally trained networks for experiments 1 and 2 take too much size and are thus not provided via this repository. Running this notebook thus requires to first run the codes Experiment_1_architecture.py AND Experiment_2_dataset_size.py (see above)**

- The **Results_model_performance.ipynb** makes a statistical analysis of the performance of the bagged 10 best models.

- The **Results_predictions.ipynb** allows generating all the other figures and the analysis presented in the paper.

All figures are saved in ./figures

### Training one network

The code Training_single.ipynb allows training only one network and playing with it.

### Getting predictions from the 10 best networks from Le Losq et al. 2021

The notebook Prediction_simple.ipynb allows to get predictions for a given melt composition. Open a Jupyter notebook server, open the notebook and follow the instructions!