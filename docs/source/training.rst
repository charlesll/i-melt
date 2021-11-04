Training the networks
=====================

Training one network
--------------------

The code `Training_single.ipynb <https://github.com/charlesll/i-melt/blob/master/Training_single.ipynb>`_ allows training only one network and playing with it. The following steps are performed.

After importing the libraries (see notebook), we load the data:

.. code-block:: python

	device = torch.device('cuda') # training on the GPU

	# custom data loader, automatically sent to the GPU
	ds = imelt.data_loader("./data/NKAS_viscosity_reference.hdf5",
                         "./data/NKAS_Raman.hdf5",
                         "./data/NKAS_density.hdf5",
                         "./data/NKAS_optical.hdf5",
                         device)

We select an architecture. For this example, we have selected the reference architecture from Le Losq et al. 2021:

.. code-block:: python

	nb_layers = 4
	nb_neurons = 200
	p_drop = 0.01


If we want to save the model and figures in the directories `./model/candidates/` and `./figures/single/`, we can use this code to check if the folders exist, and create them if not:

.. code-block:: python

	dirName = './model/candidates/'
	try:
	    # Create target Directory
	    os.mkdir(dirName)
	    print("Directory " , dirName ,  " Created ")
	except FileExistsError:
	    print("Directory " , dirName ,  " already exists")

	dirName = './figures/single/'
	try:
	    # Create target Directory
	    os.mkdir(dirName)
	    print("Directory " , dirName ,  " Created ")
	except FileExistsError:
	    print("Directory " , dirName ,  " already exists")

Now we need a name for our model, we can generate it with the hyperparameters actually, this will help us having automatic names in case we try different architectures:

.. code-block:: python

	name = "./model/candidates/l"+str(nb_layers)+"_n"+str(nb_neurons)+"_p"+str(p_drop)+"_test"+".pth"

and we declare the model using `imelt.model()`:

.. code-block:: python

	neuralmodel = imelt.model(4,nb_neurons,nb_layers,ds.nb_channels_raman,p_drop=p_drop)

We select a criterion for training (the MSE criterion from PyTorch) and send it to the GPU device

.. code-block:: python

	criterion = torch.nn.MSELoss(reduction='mean')
	criterion.to(device) # sending criterion on device

Before training, we need to initilize the bias layer using the imelt function, and we send the network parameters to the GPU:

.. code-block:: python

	neuralmodel.output_bias_init()
	neuralmodel = neuralmodel.float() # this is just to make sure we are using always float() numbers
	neuralmodel.to(device)

Training will be done with the `ADAM <https://arxiv.org/abs/1412.6980>`_ optimizer with a tuned learning rate of 0.0006:

.. code-block:: python

	optimizer = torch.optim.Adam(neuralmodel.parameters(), lr = 0.0006)

We have build a function for training in the imelt library that performs early stopping. You have to select:

* the patience (how much epoch do you wait once you notice the validation error stop improving)
* the min_delta variable, that represents the sensitivity to determine if the RMSE on the validation dataset really improved or not

The `imelt.training()` function outputs the trained model, and records of the training and validation losses during the epochs.

Training can thus be done with this code:

.. code-block:: python

	neuralmodel, record_train_loss, record_valid_loss = imelt.training(neuralmodel,ds,
	                                                                     criterion,optimizer,save_switch=True,save_name=name,
	                                                                     train_patience=400,min_delta=0.05,
	                                                                     verbose=True)

Hyperparameter tuning
---------------------

Random search
^^^^^^^^^^^^^

A Random search experiment as well as the experiment about the dataset size are done in the `Experiment_1_architecture.py <https://github.com/charlesll/i-melt/blob/master/Experiment_1_architecture.py>`_ and `Experiment_2_dataset_size.py <https://github.com/charlesll/i-melt/blob/master/Experiment_2_dataset_size.py>`_ codes. Due to the large amount of calculations, training is best done on GPU. *Training takes ~72 hours or more on a Dell Precision 5251 equipped with a RTX 4000 NVIDIA GPU.*

Bayesian optimization
^^^^^^^^^^^^^^^^^^^^^

The `Training_BO.ipynb <https://github.com/charlesll/i-melt/blob/master/Training_BO.ipynb>`_ notebook allows to perform Bayesian Optimization for hyperparameter selection using AX plateform.

Training candidates
-------------------

The code `Training_Candidates.py <https://github.com/charlesll/i-melt/blob/master/Training_candidates.py>`_ allows training 100 networks with the reference architecture and selects the 10 best ones, which are saved in ./model/best/ and used for future predictions.

*The 10 best models at the time of publication are already provided in ./model/best/. Please beware that running this notebook will update them and can affect slightly the results because each neural network training begins from a different, randomly generated starting point.*
