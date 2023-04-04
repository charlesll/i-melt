Installation
============

General preparation
-------------------

Rampy runs with a traditional Python stack.

If you are not familiar with Python, you can first have a look at the `scipy lecture notes <https://scipy-lectures.org/>`_,
a set of tutorials for the beginner.

You can install `Anaconda Python <https://www.anaconda.com/products/individual>`_ to get a running Python distribution. See the documentation of Anaconda for those steps.

Installation of libraries for i-Melt
------------------------------------

The necessary libraries are listed in the requirements.txt file.

Scripts for building, training models and providing useful functions are contained in the `src <https://github.com/charlesll/i-melt/blob/master/src/>`_ folder.

Starting from a barebone Python 3 environment with pip installed, simply open a terminal pointing to the working folder and type::

	$ pip install --user -r requirements.txt
