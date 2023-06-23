# bsc_physics_cleaner

Repository for BSc. Physics and Astronomy project: Improving the Performance of Conservative-to-Primitive Inversion in Relativistic Hydrodynamics Using Artificial Neural Networks.

## Directory structure

`models` contains trained models with their `net.pth`, `optimizer.pth`, `scheduler.pth`, `net.pt`, and all other data saved in training csv and json files. The directories also contain their own local copy of the scripts in which the hyperparameters, subparameters and file output names are set to correspond with the model in question. These local scripts are just (outdated) states of the scripts `SRHD_ML.ipynb` (or `SRHD_ML.py`) and  `GRMHD_ML.ipynb` (or `GRMHD_ML.py`) that one can find in the `src` directory; they provide the models as generated for the thesis.

`src` is the directory in which one can experiment with creating new models. It has the the most up-to-date version of the scripts for SRHD and GRMHD. The SRHD script is itself an outdated version of the GRMHD script; it can be continued to be used independently from the GRMHD script, but it has more bugs than the GRMHD script.

C++ source code files are located in the `cpp` directories.

## Using the scripts on different systems

### Using the scripts on Google Colab

Using either the SRHD or the GRMHD script on Google colab is straightforward: open the jupyter notebook file in Colab and 

1. Set `drive_folder` to save files to your desired google drive directory.
2. Comment (not uncomment) the first line of the drive mounting cell.
3. Comment (not uncomment) the first line of the `pip install` cell.
4. Follow _How to use this notebook_ at the top of the notebook as usual.

### Using the scripts on (MMAAMS) workstation

1. If there is no access to a jupyter environment, use the `.py` version of the script instead.
2. Follow _How to use this notebook_ at the top of the notebook as usual.

### Using the scripts on a local machine

1. Follow _How to use this notebook_ at the top of the notebook as usual.
