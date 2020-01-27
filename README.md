# neuravi

(c) Charles Le Losq, lelosq@ipgp.fr

License to be determined.

## REQUIREMENTS

- Python 3.7 + Jupyter stack
- pytorch
- numpy
- pandas
- scipy
- matplotlib
- mpltern
- tqdm

The model and useful functions are contained in the neuravi.py file.

## Data

All data are given in a Database_IPGP.xlsx file in the data folder. Raman spectra are contained in a ./data/raman/ folder.

## Data preparation

The notebook Dataset_preparation allows preparation of the datasets, which are subsequently saved in HDF5 format in the data folder.

The Dataset_visualization notebook show the distribution of data in ternary plots.

The preparation for Raman spectra is done in a notebook present in the data folder, see Raman_prep_python.ipynb.

## Training the networks

2,000 models were trained in the Training_experiments.ipynb notebook. The effect of the dataset size is also tested. Due to the large amount of calculations, training is best done on GPU.

Training takes 24 hours on a Dell Precision 5251 equipped with a RTX 4000 NVIDIA GPU.

The notebook Training_single_forexampleonly.ipynb allows training a single network, and is useful to play around.

## Repeating the result analysis

Analysis of training results is done in two steps.

- First, run the Results_experiments.ipynb notebook, which will allow observing the results of the experimenbts and generate already some supplementary figures. In this notebook, the 10 best networks (given validation data error) are selected and their reference saved.

- Then, run the Results_predictions.ipynb to generate all the other figures and the analysis presented in the paper.
