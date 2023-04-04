Predictions
===========

Web calculator
--------------

The easiest way to try i-Melt is to use the `web calculator <https://share.streamlit.io/charlesll/i-melt/imelt_streamlit.py>`_.

Jupyter notebooks
-----------------

More control can be achieved using directly the i-melt library. 

Several example notebooks are provided in the main repository. We invite you to have a look at them directly.

If you want to have an example of use for making predictions for a given composition, have a 
look at the `Example_Prediction_oneComposition.ipynb <https://share.streamlit.io/charlesll/i-melt/Example_Prediction_oneComposition.ipynb>`_ notebook.

The steps are simple. First, import the necessary libraries and imelt:

.. code-block:: python

  %matplotlib inline #matplotlib magic


  import numpy as np # for arrays
  import pandas as pd # for dataframes
  import matplotlib.pyplot as plt # for plotting
  import src.imelt as imelt # imelt core functions
  import src.utils as utils # utility functions

Then, we can load the pre-trained i-melt models as one Python object:

.. code-block:: python

  imelt_model = imelt.load_pretrained_bagged()

Now we can define a composition of interest in a Panda dataframe.

.. code-block:: python

  composition = pd.DataFrame()
  composition["sio2"] = 0.75
  composition["al2o3"] = 0.125
  composition["na2o"] = 0.125
  composition["k2o"] = 0.0
  composition["mgo"] = 0.0
  composition["cao"] = 0.0

We now need to add some descriptors. A utility function is provided to do so:

.. code-block:: python
  
  composition = utils.descriptors(composition).values

Predictions can be obtained for Tg with:

.. code-block:: python

  tg = imelt_model.predict("tg", composition)

We can predict the viscosity with the Vogel-Tammann-Fulscher equation. First, we create a array containing the temperatures of interest, then we calculate the viscosity:

.. code-block:: python

  T_range = np.arange(600, 1500, 1.0) # from 600 to 1500 K with 1 K steps
  viscosity = imelt_model.predict("tvf",composition*np.ones((len(T_range),39)),T_range.reshape(-1,1))

Many other predictions are possible, look at the Jupyter notebooks for more details and examples.
