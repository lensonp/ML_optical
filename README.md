
# ML_optical
A set of modules exploring scikit-learn-based ML models for predicting optical properties from structural data


## Input data
Current version runs two types of structural data: 

1. Atomic structure, encoded as distance or Coulomb matrices
2. The features computed by the [PaDel molecular descriptor code](http://padel.nus.edu.sg/software/padeldescriptor/)
which takes input from .mol structure files produced by [ChemAxon's Marvin](https://www.chemaxon.com/products/marvin/marvinsketch/)


## Training set
Absorption peaks are taken from measured spectra
reported within the [Science-SoftCon UV/Vis+ Spectra Database](http://www.science-softcon.de/)


## Instructions

### Generating input data 
Molecule structures are read in from `.xyz` structural data files
and processed into `MolStruct` (`molstruct.py`) objects.
A set of `MolStruct` objects is constructed by the `struct_data.py` module,
one `MolStruct` for each of the molecule names in the file `mols.txt`.

### Running the code
`python ML_optical.py` should do the trick.

When `ML_optical.py` is run:

1. `struct_data.py` reads `mols.txt` and builds `MolStruct` objects
2. Coulomb matrices are read from the `MolStruct` objects and stored in a `Cmat_Dtb` (`cmat_dtb.py`)
3. PaDel descriptor outputs are read in to a `PaDel_Dtb` object (`padel_dtb.py`)
4. Features are loaded from `Cmat_Dtb` and `PaDel_Dtb` objects 
5. Absorption peaks are read in from `mol_data/<molecule>/w_pk.dat`, 
	where the first line is a computed absorption peak by LDA TD-DFT, 
	and the second line is the corresponding measured absorption peak
6. The user is prompted through the application of a selection of ML models
7. Plots of training and (leave-one-out) cross-validation errors are generated from `ML_plot_routines.py`
	to aid in feature and parameter selection
 
Versions used for development:

* `scikit-learn` 
* `python`


