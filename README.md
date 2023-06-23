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
4. Follow _How to use this notebook_ at the top of the script.

### Using the scripts on (MMAAMS) workstation

1. If there is no access to a jupyter environment, use the `.py` version of the script instead.
2. Follow _How to use this notebook_ at the top of the script.

### Using the scripts on a local machine

1. Follow _How to use this notebook_ at the top of the script.

## Installation

### MMAAMS workstation

#### Installation for running model python scripts

1. Clone the repository to the desired location.

At the time of writing, Anaconda was required for a more recent python version. The version that was available on the MMAAMS workstation  downgraded PyTorch to a version incompatible with the `sm_86` architecture of the Nvidia RTX A6000 GPU of the workstation. Anaconda installs a sandboxed newer version of python such that PyTorch is not downgraded and `sm_86` architecture is supported. We hav confirmed the scripts to work well with the GPU on python version 3.11.3.

2. [Install anaconda](https://pytorch.org/get-started/locally/#linux-anaconda)

3. Setup a conda virtual environment

```sh
conda create -n <env_name> python # Create a new env with the latest python version.
#conda create -n wpa python=3.x.x # OR Create a new env with the specified python version.
```

4. Activate the environment

```sh
conda activate <env_name>
```

Note that the environment must be activated every time to run the scripts.

5. Install the required packages ([source](https://pytorch.org/))

```sh
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

6. Comment out `torch` in `<git_root>/requirements.txt` to prevent pip installation overriding the pytorch conda installation.

7. In `<git_root>`, run

```sh
pip install -r requirements.txt
```

8. Run the model in question by following _How to use this notebook_ at the top of the python script.

#### Installation for running model C++ scripts

Running a model in C++ requires libtorch. At the time of writing, we could not get libtorch to work with the `sm_86` architecture of the Nvidia RTX A6000 GPU of the workstation, and so we ran it on the CPU only. These are the installation instructions for the latter procedure.