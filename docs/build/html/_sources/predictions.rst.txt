Predictions
===========

Web calculator
--------------

The easiess way to try i-Melt is to use the [web calculator](https://share.streamlit.io/charlesll/i-melt/imelt_streamlit.py).

Jupyter notebook
----------------

More control can be achieved using directly the i-melt library. The notebook Prediction_simple.ipynb shows an example of how predictions can be done.

The steps are simple. First, import the necessary libraries and imelt:

.. code-block:: python

  %matplotlib inline #matplotlib magic


  import numpy as np # for arrays
  import matplotlib.pyplot as plt # for plotting
  import imelt # the imelt library

Then, we can load the pre-trained i-melt models as one Python object:

.. code-block:: python

  imelt_model = imelt.load_pretrained_bagged()

Now we can define a composition of interest in a 1x4 numpy array

.. code-block:: python

  composition = np.array([0.65, 0.10, 0.20, 0.05]).reshape(1,-1)

Predictions can be obtained for Tg with:

.. code-block:: python

  tg = imelt_model.predict("tg", composition)

We can predict the viscosity with the Vogel-Tammann-Fulscher equation. First, we create a array containing the temperatures of interest, then we calculate the viscosity:

.. code-block:: python

  T_range = np.arange(600, 1500, 1.0) # from 600 to 1500 K with 1 K steps
  viscosity = imelt_model.predict("tvf",composition*np.ones((len(T_range),4)),T_range.reshape(-1,1))

Many other predictions are possible, look at the [Jupyter notebook](https://github.com/charlesll/i-melt/blob/master/Prediction_simple.ipynb) for more details!
