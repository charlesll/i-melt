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

i-Melt is now a Python package. It supports Python 3.10 or higher. Install it using pip:

.. code-block:: console

	$ pip install imelt


Optional dependencies
---------------------

We simplified the dependencies of the latest version of i-Melt to avoid installation issues, particularly with aws-fortuna that still creates problems. The following packages are not anymore direct dependencies for i-Melt but may be required to run the examples/paper replicate notebooks:

- aws-fortuna
- seaborn
- mpltern >= 0.4
- uncertainties
- jupyter
- corner
- heatmatpz
- torchvision

We recommend installing them when needed. If you are running into an installation problem for those packages, unfortunately we cannot do anything so please contact directly their development team via their Github Issues page.