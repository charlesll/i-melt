Predictions
===========

Web calculator
--------------

The easiest way to try i-Melt is to use the `web calculator <https://i-melt.streamlit.app/>`_.

It has been significantly improved in version 2.2.0, and now allows you to predict all properties at once, using only one forward pass in the neural network. This is much more efficient and faster than before. You can also input the composition in mole fractions or weight percent, and the calculator will automatically convert it to the right format for the model.

Jupyter notebooks
-----------------

More control can be achieved using directly the i-melt Python library. Have a look at the :doc:`tutorials` page.

Import the model
----------------

The steps are simple. First, import imelt and other libraries as you want. Here is an example of import, using a traditional scientific Python stack:

.. code-block:: python

  %matplotlib inline #matplotlib magic


  import numpy as np # for arrays
  import pandas as pd # for dataframes
  import matplotlib.pyplot as plt # for plotting
  import imelt

Then, we can load the pre-trained i-melt models as one Python object:

.. code-block:: python

  imelt_model = imelt.load_pretrained_bagged()

Generate a query
----------------

Now we can define a composition of interest using the `generate_query_single` function:
It does everything automatically for us, including the addition of descriptors.

.. code-block:: python

  composition = imelt.generate_query_single(sio2 = 75.0, 
                                          al2o3 = 12.5,
                                          na2o = 12.5, 
                                          k2o = 0.0,
                                          mgo = 0.0,
                                          cao = 0.0, 
                                   composition_mole=True)

You can also use the `imelt.generate_query_range` function to generate a query for a range of composition. See for an example the :doc:`tutorials`.

Make a prediction
-----------------

To get predictions from the model `imelt_model`, we use its `predict` method:

  .. code-block:: python

      prediction = imelt_model.predict(properties, composition)

where properties is a list of strings indicating the properties you want, composition is a compositional array that has been put in good shape (simply use the `generate_query_single` or `generate_query_range` to do so). Optional arguments are the temperature T and lbd, the optical wavelength at which you want the optical refractive index if that is what you are after.

Here is a list of the "property" available:

  - melt viscosity (log10 Pa s) using Adam-Gibbs: enter "ag"
  - melt viscosity (log10 Pa s) using Vogel-Tammann-Fulcher: enter "tvf"
  - melt viscosity (log10 Pa s) using Free Volume: enter "cg"
  - melt viscosity (log10 Pa s) using Avramov-Milchev: enter "am"
  - melt viscosity (log10 Pa s) using MYEGA: enter "myega"
  - melt fragility : enter "fragility"
  - melt liquidus (K) : enter "liquidus"
  - glass transition temperature (K): enter "tg"
  - glass configurational entropy (J/mol/K) : enter "sctg"
  - glass density (in K): enter "density_glass"
  - glass elastic modulus (GPa) : enter "elastic_modulus"
  - glass coefficient of thermal expansion : enter "cte"
  - glass Abbe number : enter "abbe"
  - glass optical refractive index : enter "sellmeier"
  - glass Raman spectrum : enter "raman_pred"

  Note that for the melt viscosity you must provide a vector of temperature, and for the glass optical refractive you must provide a vector of wavelength.

For instance, if you want predictions for melt viscosity between 1000 and 3000 K with a step of 1 K, you will do

.. code-block:: python

    T_range = np.arange(1000.0, 3000.0, 1.0)
    predicted_props = imelt_model.predict(["vft",], composition, T_range)

`predicted_props` is a dictionary containing the properties you asked for, in this case the VFT viscosity. You can access it as:

.. code-block:: python

    vft_viscosity = predicted_props["vft"]

To get the glass optical refractive index at 589 nm, you will do:

WARNING : lambda is provided in microns !

.. code-block:: python
  
    lbd = np.array([589.0*1e-3]) # warning: enter wavenumber in microns
    predicted_props = imelt_model.predict(["sellmeier", ], composition, lbd=lbd) 
    ri = predicted_props["sellmeier"]

You could also directly query the refractive index by passign a string instead of a list:

.. code-block:: python

  ri = imelt_model.predict("sellmeier", composition, lbd=lbd)

And for a property such as Tg, you can do:

.. code-block:: python

  tg = imelt_model.predict("tg", composition)

The interest of passing a list of properties is that you can ask for several properties at once, and the model will do only one forward pass in the neural network to make all the predictions. This results in better efficiency and significantly less computing time when making multiple queries.

For instance, for all the above properties, you can do:

.. code-block:: python

  predicted_props = imelt_model.predict(["ag", "tvf", "cg", "am", "myega", 
                                         "fragility", "liquidus", "tg", 
                                         "sctg", "density_glass", 
                                         "elastic_modulus", "cte", 
                                         "abbe", "sellmeier", "raman_pred"], 
                                        composition, 
                                        T=T_range,
                                        lbd=lbd)

The predicted_props dictionary will then contain all the properties you asked for, and you can access them using their keys. For example, to get the Adam-Gibbs viscosity, Vogel-Tammann-Fulcher viscosity, and the glass transition temperature, you can do:

.. code-block:: python

  ag_viscosity = predicted_props["ag"]
  tvf_viscosity = predicted_props["tvf"]
  tg_temp = predicted_props["tg"]


Get error bars
--------------

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
  viscosity = imelt_model.predict("tvf", composition, T_range)

In the above code note that the composition array has to be modified so that you have as many lines as you have temperatures to predict.

Many other predictions are possible, look at the :doc:`tutorials` for more details and examples.
