# i-Melt

(c) 2021 Charles Le Losq, lelosq@ipgp.fr

## NEWS

### V1.1

- Version submitted after revisions for GCA manuscript
- various fixes, performance improvements, etc.
- addition of a class for storing the weights of the different loss functions
- the activation function type is passed as an optional argument to the model class
- training function as now a "save switch", allowing to turn off saving the model
- calculation of validation loss during training is now done without asking for the gradient (smaller memory footprint)
- there is also a training2() function that splits the dataset in K folds to avoid memory issues for small GPU (slower training but much smaller memory footprint)
- a function R_Raman() now allows calculating the parameter R_Raman automatically.
- The notebook for the two experiments was moved in two individual python code, easier to run on a cluster.
- notebooks for showing the results all are improved.
- a "best" architecture was selected and is now used for candidate training (4 layers, 300 neurons / layer, dropout 0.01) and selection (10 best networks are kept for predictions)
- a notebook allows simple predictions to be perform for one melt composition, see Prediction_simple.ipynb


### V1.0

- Version initially submitted to Geochimica Cosmochimica Acta.
