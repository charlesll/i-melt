i-Melt
======

(c) 2021-2025 Charles Le Losq and co., lelosq@ipgp.fr

News
====

Version 2.2.0
--------------

- All properties can now be computed using only a single forward pass in the neural network with the ``model.predict_all()`` function. This significantly reduces computing time, especially for Monte Carlo (MC) Dropout. The old API remains available, but using ``predict_all()`` is encouraged. It returns a dictionary containing all latent variables and observed properties.
- The ``predict()`` function API has changed: several properties can now be requested by passing a list to the ``methods`` argument. The old ``method`` argument still works. For multiple property predictions (latent or observed), use the ``methods`` argument. This results in only one forward pass through the neural network, improving efficiency and reducing computing time.
- The Streamlit app has been updated and is now significantly faster, thanks to the new ``predict()`` behavior. Training neural networks is also faster as viscosity predictions are grouped, yielding approximately a factor of 2 speed-up.
- All example notebooks have been updated accordingly.

Version 2.1.4
--------------

- Updated minimal Python version to 3.10 due to issues with older releases and dependencies.

Version 2.1.3
--------------

- Updated build process for PyPI deployment.

Version 2.1.1
--------------

- Removed unnecessary dependencies that were causing installation issues unrelated to i-Melt.

Version 2.1.0
--------------

- i-Melt is now a Python package: install with ``pip install imelt``!
- New examples provided in the ``examples`` folder of the repository.
- The database has been moved to ``./src/imelt/data``.
- New helper functions available: ``generate_query_single`` and ``generate_query_range`` to simplify queries.
- The results notebook has been renamed to ``./examples/Replicate_2023_paper.ipynb``.

Version 2.0.1
--------------

- Updated the results notebook ``Results_paper.ipynb`` following manuscript reviews.

Version 2.0
------------

- Added CaO and MgO components.
- Added new properties.
- Added error bar calculations.
- Updated the Streamlit online calculator.
- Numerous model changes and notebook clean-up.
- All source code relocated to ``./src``.
- Updated all files (database, models, etc.).

Version 1.2
------------

- Version submitted after 2nd round of minor revisions at *Geochimica et Cosmochimica Acta* (GCA).
- Various fixes implemented.
- Fixed the ``requirements.txt`` file.
- Added automatic creation of the ``./model/candidates/`` folder in ``Training_single.ipynb``.
- Added important clarifications to ``README.md``.

Version 1.1
------------

- Version submitted after first round of revisions for GCA manuscript.
- Various fixes and performance improvements.
- Added a class to store weights of different loss functions.
- Activation function type can be passed as an optional argument to the model class.
- Training function now includes a "save switch" to disable model saving if desired.
- Validation loss calculation during training now excludes gradient computation, reducing memory usage.
- New ``training2()`` function performs K-fold dataset splitting to avoid memory issues on small GPUs (slower, but much smaller memory footprint).
- New function ``R_Raman()`` allows automatic calculation of the R_Raman parameter.
- Notebooks for experiments have been replaced with Python scripts for easier cluster execution.
- Improved result display notebooks.
- A "best" architecture has been selected for candidate training (4 layers, 300 neurons per layer, dropout 0.01) and selection (10 best networks kept for predictions).
- A simple prediction notebook for single melt compositions is available: ``Prediction_simple.ipynb``.

Version 1.0
------------

- Initial version submitted to *Geochimica et Cosmochimica Acta* (GCA).
