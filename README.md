# DOCUMENTATION

Repository for BSc. Physics and Astronomy project: Improving the Performance of Conservative-to-Primitive Inversion in Relativistic Hydrodynamics Using Artificial Neural Networks. Thesis and presentation can be found in `thesis.pdf` and `presentation.pdf` in the root directory.

## DIRECTORY STRUCTURE

`models` contains trained models with their `net.pth`, `optimizer.pth`, `scheduler.pth`, `net.pt`, and all other data saved in csv and json files. The directories also contain their own local copy of the scripts in which the hyperparameters, subparameters and file output names are set to correspond with the model in question. **These local scripts provide the models in their states as they were generated for the thesis and are outdated** states of the scripts `SRHD_ML.ipynb` (or `SRHD_ML.py`) and  `GRMHD_ML.ipynb` (or `GRMHD_ML.py`) that are found in the `src` directory. They are outdated in having bugs that are fixed later on and in having outdated comments (see `addendum/`).

`src` is the directory in which one can experiment with creating new models. It has the the most up-to-date version of the scripts for SRHD and GRMHD. **The SRHD script is itself an outdated version of the GRMHD script**; it can continue to be used independently from the GRMHD script, but it has more bugs than the GRMHD script. A listing of commit messages between the two from the original older repository of the project can be found in `addendum/commit_messages_SRHD_to_GRMHD.txt`. For continuation of the project, we advise to just continue to edit the script `GRMHD_ML.ipynb` (or `GRMHD_ML.py`) and keep track of significant states of the script, e.g. optimizing with such and such parameters, in some other way, and to implement an easy way to load different states quickly.

C++ source code files are located in the `cpp` directories.

## INSTALLATION

### Local machine

1. Create a virtual environment in conda or python venv if desired.

#### Installation for the python scripts

2. Run

```sh
pip install -r requirements.txt
```

Make sure torch is uncommented in the file.

3. Follow _How to use this notebook_ at the top of the script in question.

#### Installation for the C++ scripts

If a GPU is available, one can follow the steps as listed under _MMAAMS workstation_, _Installation for the C++ scripts_, but choose a cuda-enabled distribution of libtorch instead, otherwise follow the same procedure.

### Colab

See _Using the scripts on Google Colab_.

### MMAAMS workstation

#### Installation for the python scripts

1. Clone the repository to the desired location.

At the time of writing (Fri Jun 23 11:17:37 AM CEST 2023), Anaconda was required to get a more recent python version running on the workstation. The version that was available on the MMAAMS workstation downgraded PyTorch to a version incompatible with the `sm_86` architecture of the Nvidia RTX A6000 GPU of the workstation. Anaconda installs a sandboxed newer version of python such that PyTorch is not downgraded in the environment and `sm_86` architecture is supported. We have confirmed the scripts to work well with the GPU on python version 3.11.3.

2. [Install anaconda](https://pytorch.org/get-started/locally/#linux-anaconda)

3. Setup a conda virtual environment

```sh
conda create -n <env_name> python
```

To setup with a specific version of python, run the following instead with `3.x.x` replaced by the desired version:

```sh
conda create -n wpa python=3.x.x
```

4. Activate the environment

```sh
conda activate <env_name>
```

Note that the environment must be activated every time to run the scripts.

5. Install the required packages ([source of the command](https://pytorch.org/))

```sh
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

6. Comment out `torch` in `requirements.txt` to prevent pip installation overriding the PyTorch installation via conda.

7. Run

```sh
pip install -r requirements.txt
```

8. Run the script for the model in question by following _How to use this notebook_ at the top of the python script.

#### Installation for the C++ scripts

Running a model in C++ requires libtorch. At the time of writing (Fri Jun 23 11:17:52 AM CEST 2023), we could not get libtorch to work with the `sm_86` architecture of the Nvidia RTX A6000 GPU on the workstation, and so we ran it on the CPU only. These are the installation instructions for the latter procedure.

1. Download libtorch into the desired directory:

```sh
wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.0.1%2Bcpu.zip
```

2. Unzip the downloaded zip file.

The next step requires the `CMakeLists.txt` file to be set up, which we have already done for all scripts. However, if problems are encountered, consult [the pytorch documentation on using torch in C++](https://pytorch.org/cppdocs/installing.html).

3. As found in the _How to use this notebook_ section in the scripts, building can be done with

```sh
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch/ ..,
cmake --build . --config release
```

The executable can then be run with `./<executable name>`.

## USING THE PYTHON SCRIPTS

### Running the python on Google Colab

Using either the SRHD or the GRMHD script on Google colab is straightforward: open the jupyter notebook file in Colab and 

1. Set `drive_folder` to save files to your desired google drive directory.
2. Comment (not uncomment) `%%script echo skipping` of the drive mounting cell.
3. Comment (not uncomment) `%%script echo skipping` line of the `pip install` cell.
4. The rest is the same as running locally, i.e. as in _Using the scripts on a local machine_.

### Running the scripts on (MMAAMS) workstation

1. If there is no access to a jupyter environment, use the `.py` version of the script instead.
2. Follow _How to use this notebook_ at the top of the script.

### Running the scripts on a local machine

1. Follow _How to use this notebook_ at the top of the script.

### Evaluating an ANN model

Evaluating of an artificial neural network model can be done with `torch.cuda.Event`. This is illustrated `model/NNGR{1,2}/NNGR{1,2}_evaluation.py`. The relevant code is:

```python
```

## TROUBLESHOOTING: INSTALLATION AND RUNNING

### Running models trained on the GPU on the CPU and vice versa

Problems can arise from running models that are trained on GPU on the CPU and vice versa. These problems are easily solved by mapping the model to the CPU or to the GPU when it is loaded. This mapping is most easily done in the python script from which the model was generated, e.g. `GRMHD_ML.ipynb` or `GRMHD_ML.py`, with e.g. the code

```python
if torch.cuda.is_available():
    net_loaded.load_state_dict(torch.load("net.pth"))
else: 
    # Map the loaded network to the CPU.
    net_loaded.load_state_dict(torch.load("net.pth", map_location=torch.device('cpu')))
```

in the _Loading_ section of the script in question.

### Running .py python files without having jupyter installed

Make sure that the  `get_ipython` lines in the `.py` files are commented out when running these files on a system without jupyter installed.