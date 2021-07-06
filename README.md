# i-Melt

(c) 2021 Charles Le Losq, lelosq@ipgp.fr

## LICENSE

Any material in this repository is under the MIT licence

See MIT license file for details

## REQUIREMENTS

see requirements.txt file

The model and useful functions are contained in the imelt.py file.

## Data

All data are given in a Database_IPGP.xlsx file in the data folder. Raman spectra are contained in a ./data/raman/ folder.

## Data preparation

The notebook **Dataset_preparation.ipynb** allows preparation of the datasets, which are subsequently saved in HDF5 format in the data folder.

The **Dataset_visualization.ipynb** notebook shows the distribution of data in ternary plots.

## Training the networks

### Hyperparameter tuning

- A Random search experiment as well as the experiment about the dataset size are done in the **Training_experiments.ipynb** notebook. Due to the large amount of calculations, training is best done on GPU. *Training takes ~72 hours or more on a Dell Precision 5251 equipped with a RTX 4000 NVIDIA GPU.*

- The **Training_BO.ipynb** notebook allows to perform Bayesian Optimization for hyperparameter selection using AX plateform.

### Training candidates

The notebook **Training_Candidates.ipynb** allows training 50 networks with the selected architecture and selects the 10 best, which are saved in ./model/best/ and used for future predictions.

## Repeating the result analysis

Analysis of results and predictions from the 10 best trained networks is done in several steps:

- The **Results_experiments.ipynb** notebook allows observing the results of the random search and dataset size experiments, and alows generating Supplementary Figure 1. 

- The **Results_model_performance.ipynb** makes a statistical analysis fo the performance fo the bagged 10 best models.

- The **Results_predictions.ipynb** allows generating all the other figures and the analysis presented in the paper.


