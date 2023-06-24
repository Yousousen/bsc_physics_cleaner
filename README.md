# DOCUMENTATION

Repository for BSc. Physics and Astronomy project: Improving the Performance of Conservative-to-Primitive Inversion in Relativistic Hydrodynamics Using Artificial Neural Networks. Thesis and presentation can be found in `thesis.pdf` and `presentation.pdf` in the root directory.

## DIRECTORY STRUCTURE

`models` contains trained models with their `net.pth`, `optimizer.pth`, `scheduler.pth`, `net.pt`, and all other data saved in csv and json files. The directories also contain their own local copy of the scripts in which the hyperparameters, subparameters and file output names are set to correspond with the model in question. **These local scripts provide the models in their states as they were generated for the thesis and are outdated** states of the scripts `SRHD_ML.ipynb` (or `SRHD_ML.py`) and  `GRMHD_ML.ipynb` (or `GRMHD_ML.py`) that are found in the `src` directory. They are outdated in having bugs that are fixed later on (see `addendum/`) and in having outdated comments.

`src` is the directory in which one can experiment with creating new models. It has the the most up-to-date version of the scripts for SRHD and GRMHD. **The SRHD script is itself an outdated version of the GRMHD script**; it can continue to be used independently from the GRMHD script, but it has more bugs than the GRMHD script. A listing of commit messages between the two from the original older repository of the project can be found in `addendum/commit_messages_SRHD_to_GRMHD.txt`. For continuation of the project, we advise to just continue to edit the script `GRMHD_ML.ipynb` (or `GRMHD_ML.py`) and keep track of significant states of the script, e.g. optimizing with such and such model with such and such parameters, in some other way, and to implement code to easily load different states quickly.

C++ source code files are located in the `cpp` directories.

## INSTALLATION

### Local machine

1. Clone the repository to the desired location.

2. Create a virtual environment in conda or python venv if desired.

#### Installation for the python scripts

3. Run

```sh
pip install -r requirements.txt
```

Make sure torch is uncommented in the file.

4. Follow _How to use this notebook_ at the top of the script in question.

#### Installation for the C++ scripts

If a GPU is available, one can follow the steps as listed under _MMAAMS workstation_, _Installation for the C++ scripts_, but choose a cuda-enabled distribution of libtorch instead. The rest of the procedure is the same.
### Colab

1. Open the github repository in Google Colab.

2. Create a copy of the jupyter notebook file that one wants to run so that one is able to save changes.

3. Continue with _Using the scripts on Google Colab_.

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

1. Download libtorch into the desired directory ([see pytorch documentation for the latest version](https://pytorch.org/cppdocs/installing.html))

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

### Running the scripts on Google Colab

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

### Loading models without training or optimizing with the .py files instead of the jupyter notebooks

In the jupyter notebooks, one can load a model without retraining and without optimizing by following _Loading an already trained model_ and _Generating the C++ model_ of _How to use this notebook_ at the top of the script. If jupyter notebook is not available, one can still follow these instructions, and one should simply explicitly comment out code that should not be run according to these instructions in the `.py` file version of the script.

### Evaluating the running time of an ANN model

Evaluating of an artificial neural network model can be done with `torch.cuda.Event`. This is illustrated at the end of `model/NNGR1/NNGR1_evaluation.py`. The relevant code is:

```python
example_input = generate_input_data(*generate_samples(1))

# Ensure that your model and input data are on the same device (GPU in this case)
model = net_loaded.to('cuda')
input_data = example_input.to('cuda')
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
output = model(input_data)
end_event.record()
torch.cuda.synchronize() # Wait for the events to be recorded

print(f"Evaluation time: {start_event.elapsed_time(end_event)} milliseconds")
```

### Other scripts in the model and src directories

Some models have scripts ending in `_train.py` and `_optimization.py`. These are just scripts in which `OPTIMIZE` is set to `False` and to `True` respectively, and, together with other settings and hyperparameters set as can be found in the file, are used to test the training or optimization process of an arbitrary model easily on the workstation without having to open the same file and editing it many times to switch between no optimization and optimization.

## TROUBLESHOOTING

### Running models, trained on the GPU, on the CPU, and vice versa

Problems can arise from running models that are trained on GPU on the CPU and vice versa. These problems are solved by mapping the model to the CPU or to the GPU when it is loaded, which can be done without retraining the model. This mapping is most easily done in the python script from which the model was generated, e.g. `GRMHD_ML.ipynb` or `GRMHD_ML.py`. To map to the CPU when CUDA is not available, one should add the following code directly after the initialization of the `net_loaded` object in the _Loading_ section of the script in question:

```python
# ...

if torch.cuda.is_available():
    net_loaded.load_state_dict(torch.load("net.pth"))
else: 
    # Map the loaded network to the CPU.
    net_loaded.load_state_dict(torch.load("net.pth", map_location=torch.device('cpu')))

# Load the optimizer from the .pth file
if torch.cuda.is_available():
  optimizer_loaded_state_dict = torch.load("optimizer.pth")
else:
  optimizer_loaded_state_dict = torch.load("optimizer.pth", map_location=torch.device('cpu'))

# Load the scheduler from the .pth file
if torch.cuda.is_available():
  scheduler_loaded_state_dict = torch.load("scheduler.pth")
else:
  scheduler_loaded_state_dict = torch.load("scheduler.pth", map_location=torch.device('cpu'))

# ... 
```

 It could be that one needs to map these state dictionaries to the CPU even if CUDA is in fact available. In that case we can simply replace the above if-else statements with the statements in the else-cases only:

```python
# ...
net_loaded.load_state_dict(torch.load("net.pth", map_location=torch.device('cpu')))
optimizer_loaded_state_dict = torch.load("optimizer.pth", map_location=torch.device('cpu'))
scheduler_loaded_state_dict = torch.load("scheduler.pth", map_location=torch.device('cpu'))
# ... 
```

On systems such as the MMAAMS workstation where CUDA is in fact available but one wants to explicitly map to the CPU, the `device` variable requires to be mapped also. This should be done before the mapping of the state dictionaries disussed above. Mapping `device` makes sure that `net_loaded` uses the correct device and that the input tensor that is used to trace the model and to then generate the `net.pt` file is mapped to correct device also. To map `device` to the CPU, one should add **before** the code in the _Loading_ section

```python
net_loaded = Net(
    n_layers_loaded,
    n_units_loaded,
    hidden_activation_loaded,
    output_activation_loaded,
    dropout_rate_loaded
).to(device)
```

, the line

```python
device = torch.device("cpu")
```

Do not forget to save and run the python script after the changes discussed have been implemented so that the `net.pt` file is updated accordingly.

### Running .py python files without having jupyter installed

Make sure that the `get_ipython` lines in the `.py` files are commented out when running these files on a system without jupyter installed.

### python vs python3

It is advised to use `python3` command instead of `python` command for running the scripts e.g. on the workstation, as `python` can still be linked to version 2 of the language.

### error loading the model

1. To resolve `error loading the model`, ensure that the `net.pt` file (not the `net.pth` file) is located in the directory specified by the `path_to_model` variable in the C++ script. `path_to_model` should include the file name itself. Note that if one specifies a relative path in `path_to_model`, this path should point to the location of `net.pt` **relative to the executable**, not relative to the source file.

2. If the error persists it is likely due to trying to load a GPU-trained model on the CPU or vice versa (see _Running models, trained on the GPU, on the CPU, and vice versa_). For instance, if `std::cerr << e.what() << '\n';` outputs

```sh
[username@pc build]$ cmake --build . --config release && ./GRMHD
[ 50%] Building CXX object CMakeFiles/GRMHD.dir/GRMHD.cpp.o
[100%] Linking CXX executable GRMHD
[100%] Built target GRMHD
error loading the model, did you correctly set the path to the net.pt file?
error: Could not run 'aten::empty_strided' with arguments from the 'CUDA' backend. 

# ...

frame #26: torch::jit::load(std::string const&, c10::optional<c10::Device>, bool) + 0xac (0x7fc32e1d7c7c in /path/to/libtorch/lib/libtorch_cpu.so)
frame #27: main + 0xb6 (0x5573f837482a in ./GRMHD)
frame #28: <unknown function> + 0x23850 (0x7fc328c39850 in /usr/lib/libc.so.6)
frame #29: __libc_start_main + 0x8a (0x7fc328c3990a in /usr/lib/libc.so.6)
frame #30: _start + 0x25 (0x5573f8374435 in ./GRMHD)
```

, then this could be caused by the problem of trying to run a GPU-trained model on the CPU. To resolve the issue one should make the modifications as specified in _Running models, trained on the GPU, on the CPU, and vice versa_.

### Issues with the STUDY_NAME constant

The file pointed to by the constant `STUDY_NAME` is a pickle file that saves all trials of an Optuna study. Some parts of the code do not run if the file specified by `STUDY_NAME` is not found. If there was no previous Optuna study, or it is unnecessary to load such an Optuna study again, `STUDY_NAME` can simply be set to `None` in order to run the code.

### save_file being undefined

The function `save_file` exists to save files to a specified google drive location; this is e.g. useful on colab where the runtime which contains saved files is automatically deleted after a period of inactivity. It is required to load the definition of `save_file` in the script even when colab or google drive is not used to save the file. If colab or google drive is not used, then `save_file` does nothing.

### NameError: name 'Net' is not defined. Did you mean: 'set'?

This issue arises when class `Net` is not defined. This class definition still needs to be known e.g. when loading a pre-trained model. The issue is resolved by having class `Net` be defined.

### NameError: name 'net' is not defined. Did you mean: 'Net'?

This issue can e.g. arise when trying to load a model without training or optimizing. Note that the `net` object is an instance of class `Net` that is only used in optimizing and training; when we load a model without either training or optimizing, we use the `net_loaded` object instead. Likewise, all the variables associated with `net`, such as `train_metrics`, `test_metrics`, `var_dict`, etc. are suffixed by `_loaded` when we load a model without optimizing or training it beforehand: `train_metrics_loaded`, `test_metrics_loaded`, `var_dict_loaded` etc. This was done so that loading of a model could be done without overriding the original variables by redefining them upon loading of a model and so that correct loading can be verified.
