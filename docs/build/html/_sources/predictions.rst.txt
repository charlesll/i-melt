Predictions
===========

Web calculator
--------------

The easiest way to try i-Melt is to use the `web calculator <https://i-melt.streamlit.app/>`_.

Jupyter notebooks
-----------------

More control can be achieved using directly the i-melt library. 

Several example notebooks are provided in the main repository. We invite you to have a look at them directly.

If you want to have an example of use for making predictions for a given composition, have a 
look at the `Example_Prediction_OneComposition.ipynb <https://share.streamlit.io/charlesll/i-melt/Example_Prediction_OneComposition.ipynb>`_ notebook.

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
We also automatically add descriptors to the composition.

.. code-block:: python

  composition = [0.75, # SiO2
                0.125, # Al2O3
                0.125, # Na2O
                0.0, # K2O
                0.0, # MgO
                0.0] # CaO

  # we transform composition in a dataframe and add descriptors
  composition = pd.DataFrame(np.array(composition).reshape(1,6), columns=["sio2","al2o3","na2o","k2o","mgo","cao"])
  composition = utils.descriptors(composition.loc[:,["sio2","al2o3","na2o","k2o","mgo","cao"]]).values

Predictions can be obtained for Tg with:

.. code-block:: python

  tg = imelt_model.predict("tg", composition)

If you want error bars, you need to ask for samples:

.. code-block:: python

  tg = imelt_model.predict("tg", composition, sampling=True, n_sample=20)
  
Here tg contains 20 samples from the 10 different models, so a total of 200 predictions. You can now calculate the standard deviation and mean values of tg as:

.. code-block:: python

  tg_standard_deviation = np.std(tg)
  tg_mean = np.mean(tg)

Another way, better, may be to ask for the 95% confidence intervals and the median:

.. code-block:: python

  tg_95CI = np.percentile(tg, [2.5, 97.5])
  tg_median = np.median(tg)

We can predict the viscosity with the Vogel-Tammann-Fulscher equation. First, we create a array containing the temperatures of interest, then we calculate the viscosity:

.. code-block:: python

  T_range = np.arange(600, 1500, 1.0) # from 600 to 1500 K with 1 K steps
  viscosity = imelt_model.predict("tvf",composition*np.ones((len(T_range),39)),T_range.reshape(-1,1))

In the above code note that the composition array has to be modified so that you have as many lines as you have temperatures to predict.

Many other predictions are possible, look at the Jupyter notebooks for more details and examples.
