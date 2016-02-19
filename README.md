
# ML_optical

A set of modules exploring scikit-learn-based ML models for predicting optical properties from structural data.
Designed around structural/optical data, 
but can be used for arbitrary inputs and targets
by loading data from files `ML_Xvals.dat` and `ML_yvals.dat`.
Example data sets are included.

### Fast instructions

Run with `python ML_optical.py`.

Input molecular data is in the `mol_data` directory, 
or anonymous data can be read in from `ML_Xvals.dat` and `ML_yvals.dat`.

### Notes 

The main run script for the code is `ML_optical.py`. 
Much of the functionality can be adjusted by editing this script.

These modules were a working part of a larger research project.
They are meant to streamline the testing and validation of models
for dozens of relatively small but high-dimensional data sets
with minimal reproduction of code.
New models can be added by importing them to the `regress.py` module 
and then writing a build case for that method in 
`regress.py`'s `build_regressor()` function.
This level of completeness
leaves the code flexible for expansion and/or re-purposing.

In previous versions, these modules included
the full range of regression techniques available in scikit-learn
including the development version (for multilayer perceptron models),
unsupervised feature exploration (PCA, clustering),
supervised classification (SVM, logistic regression, GDA),
and a module of custom kernels specific to structural features.
I left these other functionalities 
out of this version to simplify the code.

## Code structure

### Script
The main run script is `ML_optical.py`. 

### Classes
 1. `MolStruct` (`molstruct.py`) - loads and operates on molecule structure data
 2. `Cmat_Dtb` (`cmat_dtb.py`) - operates on Coulomb matrix data 
 3. `PaDel_Dtb` (`padel_dtb.py`) - operates on PaDel molecule descriptor data 

Note: This version of the code could have `cmat_dtb.py` and `padel_dtb.py` as modules.
They are classes because, in previous (and maybe future) versions, 
multiple distinct databases would exist at one time.

### Modules
 1. `struct_data.py` - builds and operates on `MolStruct` objects for all molecules in `mols.txt`
 2. `ML_plot_routines.py` - plotting routines for visualizing ML results
 3. `regress.py` - a module for interfacing with a variety of `scikit-learn`-based regression models

Note: Addition of new ML models can be accomplished by adding a build method (2 lines) to `regress.py`
and adding flags and parameters to the beginning of `ML_optical.py` 

## Data set

### Input structural features or other user-specified data
These modules load and operate on three types of data: 

1. Atomic structure, encoded as Coulomb matrices (`struct_data.py`, `cmat_dtb.py`)
2. The features computed by the [PaDel molecular descriptor code](http://padel.nus.edu.sg/software/padeldescriptor/)
	which takes input from .mol structure files produced by [ChemAxon's Marvin](https://www.chemaxon.com/products/marvin/marvinsketch/) 
	(`padel_dtb.py`)
3. Anonymous inputs from a file, `ML_Xvals.dat`, containing a matrix of inputs, 
where each row of the file is a feature vector of the sample.

### Target optical or user-specified data
For optical data, the `struct_data.py` module includes
code for loading absorption peaks read approximately
from publications found in the 
[Science-SoftCon UV/Vis+ Spectra Database](http://www.science-softcon.de/),
as well as the (attempted) corresponding peaks obtained 
by post-processing an LDA TD-DFT computation.
The user has several options for targets when running `ML_optical.py`:

1. The absorption peak frequencies, computed or measured
2. The error between the computed and measured peak frequencies
3. An anonymous data set loaded by reading a file `ML_yvals.dat`

Note:  
A choice of absorption peak index is prompted for when working with the absorption data.
The code will work for the 0th through the 5th absorption peaks, 
but the data sets are very sparse for peaks 2-5.
The code will crash for peak 6, because none of the spectra used here have more than 6 peaks. 

## Instructions

### Generating input data 

To use arbitrary input data,
the file `ML_Xvals.dat` can be filled with a 
(`n_samples` by `n_features`) matrix of floating-point values,
and the file `ML_yvals.dat` can be loaded with a list of target values.
Both files should have no headers or additional information, 
only floating point numbers.

By default, when `ML_optical.py` is run,
molecule structures are read in from `.xyz` structural data files
and processed into `MolStruct` objects (`molstruct.py`).
A set of `MolStruct` objects is constructed by the `struct_data.py` module,
one `MolStruct` for each of the molecule names in the file `mols.txt`.

The Coulomb matrices from the `MolStruct` objects 
are loaded into a `Cmat_Dtb` object (`cmat_dtb.py`) 
which performs further operations on the set of Coulomb Matrices
to form sensible feature-vector inputs from them.

PaDel feature dictionaries from `mol_data/<mol>/<mol>.pdl`
are loaded into a `PaDel_Dtb` object (`padel_dtb.py`).
One PaDel dictionary is loaded for each molecule in `mols.txt`.
Then, the dictionary is reduced to contain only the set of features
that are real valued over the entire data set,
which are then used as feature-vector inputs.

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
6. The user is prompted through the selection of features and targets 
	(precise control over the models and parameters is best accomplished by editing the first few lines of `ML_optical.py`) 
7. Plots of training and (leave-one-out) cross-validation errors are generated using `ML_plot_routines.py`
 
### Versions used for development
* `scikit-learn` 0.17 
* `python` 2.7.6

### References
The works used to estimate absorption peak data
can all be found in the included `ML_optical.bib` file 

