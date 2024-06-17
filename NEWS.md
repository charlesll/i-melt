# i-Melt

(c) 2021-2024 Charles Le Losq and co., lelosq@ipgp.fr

## NEWS

### V2.1.0

- i-Melt is now a Python package : install it using 'pip install imelt' !
- new examples, see the new examples folder in the repository
- the database has been moved in the folder ./src/imelt/data
- functions to simplify queries are available: generate_query_single and generate_query_range

### V2.0.1

- update of the Results_paper.ipynb notebook following the reviews of the manuscript.

### V2.0

- addition of CaO and MgO
- addition of new properties
- error bar calculations
- update of the streamlit online calculator
- many model changes, cleaning the notebooks
- all source code is now in ./src
- update of all files (database, models, etc.)

### V1.2

- Version submitted after 2nd round of minor revisions at GCA
- Various fixes
- Fixed the requirements.txt file
- Added automatic ./model/candidates/ folder creation in Training_single.ipynb
- Added various important precisions to README.md

### V1.1

- Version submitted after revisions for GCA manuscript
- Various fixes, performance improvements, etc.
- Addition of a class for storing the weights of the different loss functions
- The activation function type is passed as an optional argument to the model class
- Training function as now a "save switch", allowing to turn off saving the model
- Calculation of validation loss during training is now done without asking for the gradient (smaller memory footprint)
- There is also a training2() function that splits the dataset in K folds to avoid memory issues for small GPU (slower training but much smaller memory footprint)
- A function R_Raman() now allows calculating the parameter R_Raman automatically.
- The notebook for the two experiments was moved in two individual python code, easier to run on a cluster.
- Notebooks for showing the results all are improved.
- A "best" architecture was selected and is now used for candidate training (4 layers, 300 neurons / layer, dropout 0.01) and selection (10 best networks are kept for predictions)
- A notebook allows simple predictions to be perform for one melt composition, see Prediction_simple.ipynb

### V1.0

- Version initially submitted to Geochimica Cosmochimica Acta (GCA).