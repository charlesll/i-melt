Database
========

All data are given in a `Database.xlsx <https://github.com/charlesll/i-melt/blob/master/data/Database.xlsx>`_ file in the `/src/imelt/data folder <https://github.com/charlesll/i-melt/tree/master/data>`_.

The data used for training the currently provided models are in HDF5 format in the folder /src/imelt/data/.

The raw Raman spectra are available in the ./data/raman/ folder.

Preparation
----------------

To prepare the data, we used several scripts described below. Beware that this was done on our machines and may not run out of the box due to folder tree structure.

The script `Dataset_preparation.py <https://github.com/charlesll/i-melt/blob/master/src/Dataset_preparation.py>`_ allows preparing the datasets, which are subsequently saved in HDF5 format in a ./data folder.

The `Dataset_visualization.py <https://github.com/charlesll/i-melt/blob/master/src/Dataset_visualization.py>`_ script allows running the generation of several figures, saved in /figures/datasets/ . Processed Raman spectra are also shown in /figures/datasets/raman/ .
