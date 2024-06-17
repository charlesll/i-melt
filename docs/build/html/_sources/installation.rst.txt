Installation
============

General preparation
-------------------


i-Melt runs with a traditional Python stack.

If you are not familiar with Python, you can first have a look at the `scipy lecture notes <https://scipy-lectures.org/>`_,
a set of tutorials for the beginner.

You can install `Anaconda Python <https://www.anaconda.com/products/individual>`_ to get a running Python distribution. See the documentation of Anaconda for those steps.

Installation of i-Melt
----------------------

i-Melt is now a Python package. It supports Python 3.8 or higher. Install it using pip:

	`$ pip install imelt`

There may be a problem with the installation of aws-fortuna.

Apparently there is problems with the versions of jax and flax for this package. I reported it but this is still ongoing.
For now, an easy fix is to install aws-fortuna, and then to upgrade jax and flax to the latest versions:

	`$ pip install --upgrade jax flax`

There will be a version error message, but disregard it, aws-fortuna works and there is no problem anymore.
