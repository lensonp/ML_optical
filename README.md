
# ML_optical

A set of modules exploring scikit-learn-based ML models for predicting optical properties from structural data.
Designed around structural/optical data, 
but can be used for arbitrary inputs and targets
by loading data from files `ML_Xvals.dat` and `ML_yvals.dat`.
Example data sets are included.


### Notes 

The main run script for the code is `ML_optical.py`. 
Much of the functionality can be adjusted by editing this script.
While it would be possible to refine the user interface 
so that one would never need to edit the main run script,
this would result in a lot of command-line I/O
with diminishing productivity returns.

These modules were a working part of a larger research project.
They are meant to streamline the testing and validation of models
for dozens of relatively small but high-dimensional data sets
with minimal reproduction of code.
New models can be added by importing them to the `regress.py` module 
and then writing a build case for that method in 
`regress.py`'s `build_regressor()` function.
This level of completeness
leaves the code flexible for expansion and/or re-purposing.

The full set of modules from this project interfaces with
the full range of regression techniques available in scikit-learn
and includes unsupervised feature exploration (PCA, clustering)
and supervised classification (SVM, logistic regression, GDA).
I left these other functionalities 
out of this version to simplify the code.

## Data set

### Input structural features or other user-specified data
These modules load and operate on three types of data: 

1. Atomic structure, encoded as distance or Coulomb matrices (`struct_data.py`, `cmat_dtb.py`)
2. The features computed by the [PaDel molecular descriptor code](http://padel.nus.edu.sg/software/padeldescriptor/)
which takes input from .mol structure files produced by [ChemAxon's Marvin](https://www.chemaxon.com/products/marvin/marvinsketch/) 
(`padel_dtb.py`)
3. Anonymous inputs from a file, `ML_Xvals.dat`, containing a matrix of inputs, 
where each row of the file is a feature vector of the sample.

### Target optical or user-specified data
For optical data, the `struct_data.py` module includes
code for loading absorption peaks that were taken from measured spectra
reported within the [Science-SoftCon UV/Vis+ Spectra Database](http://www.science-softcon.de/),
as well as the (attempted) corresponding peaks obtained 
by post-processing an LDA TD-DFT computation.
The user has several options for targets when running `ML_optical.py`:

1. The absorption peak frequencies, computed or measured
2. The error between the computed and measured peak frequencies
3. An anonymous data set loaded by reading a file `ML_yvals.dat`

Note:  
Very little explicit checking is performed on the input data.
The code may throw errors on poorly formed inputs. 


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
and processed into `MolStruct` (`molstruct.py`) objects.
A set of `MolStruct` objects is constructed by the `struct_data.py` module,
one `MolStruct` for each of the molecule names in the file `mols.txt`.

The Coulomb matrices from the `MolStruct` objects 
are loaded into a Coulomb matrix database (`cmat_dtb.py`) 
which performs further operations on the set of Coulomb Matrices
to form sensible feature-vector inputs from them.

PaDel feature dictionaries are saved in `mol_data/<mol>/<mol>.pdl`.
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
	(control over the models and parameters is best accomplished by editing `ML_optical.py`) 
7. Plots of training and (leave-one-out) cross-validation errors are generated from `ML_plot_routines.py`
 
Versions used for development:
* `scikit-learn` 
* `python`


