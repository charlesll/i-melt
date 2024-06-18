Training
=====================

The i-Melt 2.1 library is meant to provide trained models and use them for predictions. 

However, if you want to play with the code and train new models, you can do so following the instructions listed below. Note that paths will probably have to be slightly modified, as the current library is intended to be used for predictions and the code for training has been written prior to the latest "production release".

Simple example
--------------

The notebook `Training_single.ipynb <https://github.com/charlesll/i-melt/blob/main/examples/Training_single.ipynb>`_ provides a simple example of how you can train your own i-Melt neural networks.

Training scripts
-------------------

Scripts for building, training models and providing useful functions are provided `here <https://github.com/charlesll/i-melt/blob/master/src/>`_.

The easiest way of training one or multiple neural networks is to use those scripts. I suggest getting a copy of the Github repository and working in it directly, it will simplify things.



Training one network
--------------------

The code `Training_single.py <https://github.com/charlesll/i-melt/blob/master/src/Training_single.py>`_ allows training only one network and playing with it. The following steps are performed.

After importing the libraries (see notebook), we load the data:

.. code-block:: python

	device = torch.device('cuda') # training on the GPU

	# custom data loader, automatically sent to the GPU
	ds = imelt.data_loader(device=device)

We select an architecture. For this example, we have selected the reference architecture from Le Losq et al. 2021:

.. code-block:: python

	nb_layers = 4
	nb_neurons = 200
	p_drop = 0.10 # we increased dropout here as this now works well with GELU units

If we want to save the model and figures in a directory such as `./outputs/`, we can use this code to check if the folder exists and create it if not:

.. code-block:: python

	imelt.create_dir('./outputs/')
	
Now we need a name for our model, we can generate it with the hyperparameters actually, this will help us having automatic names in case we try different architectures:

.. code-block:: python

	name = "./outputs/candidates/l"+str(nb_layers)+"_n"+str(nb_neurons)+"_p"+str(p_drop)+"_test"+".pth"

and we declare the model using `imelt.model()`:

.. code-block:: python

	neuralmodel = imelt.model(ds.x_visco_train.shape[1], # input shape
							hidden_size=nb_neurons, # number of neurons per hidden layer
							num_layers=nb_layers, # number of hidden layers
							nb_channels_raman=ds.nb_channels_raman, # number of input channels for Raman spectra
							activation_function = torch.nn.GELU(), # ANN activation function
							p_drop=p_drop # dropout probability
							)

We select a criterion for training (the MSE criterion from PyTorch) and send it to the GPU device

.. code-block:: python

	criterion = torch.nn.MSELoss(reduction='mean')
	criterion.to(device) # sending criterion on device

Before training, we need to initilize the bias layer using the model `output_bias_init` method, and we send the network parameters to the GPU:

.. code-block:: python

	neuralmodel.output_bias_init()
	neuralmodel = neuralmodel.float() # this is just to make sure we are using always float() numbers
	neuralmodel.to(device)

Training will be done with the `ADAM <https://arxiv.org/abs/1412.6980>`_ optimizer with a tuned learning rate of 0.0003:

.. code-block:: python

	optimizer = torch.optim.Adam(neuralmodel.parameters(), lr = 0.0003)

We have build a function for training in the imelt library that performs early stopping. You have to select:

* the patience (how much epoch do you wait once you notice the validation error stops improving).
* the min_delta variable, which represents the sensitivity to determine if the RMSE on the validation dataset really improved or not.

The `imelt.training()` function outputs the trained model, and records of the training and validation losses during the epochs.

Training can thus be done with this code:

.. code-block:: python

	neuralmodel, record_train_loss, record_valid_loss = imelt.training(neuralmodel, # model
                                                                   ds, # dataset
                                                                   criterion, # criterion for training (RMSE here)
                                                                   optimizer, # optimizer: ADAM
                                                                   save_switch=True, # do we save the best models?
                                                                   save_name=name, # where do we save them?
                                                                   train_patience=250, # how many epochs we wait until early stopping?
                                                                   min_delta=0.05, # how sensitive should we be to consider the validation metric has improved?
                                                                   verbose=True # do you want text?
                                                                   )

Hyperparameter tuning
---------------------

RAY TUNE + OPTUNA
^^^^^^^^^^^^^^^^^

In the version 2.0 and above, we rely on `Ray Tune <https://docs.ray.io/en/latest/tune/index.html>`_ and `Optuna <https://optuna.org/>`_ to search for the best models.

The script `ray_opt.py <https://github.com/charlesll/i-melt/blob/master/src/ray_opt.py>`_ allows running a Ray Tune experiment.

The script `ray_select.py <https://github.com/charlesll/i-melt/blob/master/src/ray_select.py>`_ allows selecting the best models 
based on posterior analysis of the Ray Tune experiment (all metrics recorded in an Excel spreadsheet that must be provided for model selection).

Training candidates
-------------------

**Note : this was used in v1.2 for model selection, but now we rely on the Ray Tune + Optuna run to select models.**

In any case, this still works. The code `Training_Candidates.py <https://github.com/charlesll/i-melt/blob/master/Training_candidates.py>`_ allows training 100 networks with a given architecture and selects the 10 best ones, which are saved in ./model/best/ and used for future predictions.
