# Supplementary Materials for Bayesian Penalized Transformation Models

## Set Up

To run the demos, you need the following available:

1. A working installation of Python 3.13 or newer
2. A Python environment, ideally a virtual environment, with the following packages installed:
    - `liesel==0.4.2`
    - `liesel_ptm==0.1.0`
    - `ipykernel` (for running a jupyter notebook)
    - `jupyter` (for running a jupyter notebook)

Sometimes the installation of Jax, a key dependency of both liesel and liesel_ptm is
tricky. For help, consider: https://docs.jax.dev/en/latest/installation.html

### MacOS Setup

#### Python 3.13

To check you Python version, run the follwing in a terminal:

```
python3.13 --version
```

If that fails, install via Python 3.13 via Homebrew (see https://brew.sh):

```
brew install python@3.13
```

Then verify success by running:

```
python3.13 --version
```

#### Virtual Environment Setup


First, make sure the working directory is the project directory:

```
pwd # prints working directory
```

If that is not the case, change to the project directory

```
cd /path/to/ptm-supplement
```

Now initialize a new virtual environment:

```
python3.13 -m venv .venv
```

Now activate the virtual environment:

```
source .venv/bin/activate
```


#### Install packages

Now install the required packages via:

```
pip install liesel liesel_ptm ipykernel jupyter
```

#### Launch Jupyter Notebook

Now you can launch the demo notebooks:

```
jupyter notebook demos/dutch-growth-study.ipynb
```

```
jupyter notebook demos/framingham-heart-study.ipynb
```


### Windows Setup

#### Python 3.13

To check your Python version, open **Command Prompt (cmd.exe)** or **PowerShell** and run:

```
py -3.13 --version
```

If that fails, try:

```
python --version
```

If neither works, download and install Python 3.13 from [python.org](https://www.python.org/downloads/).  
During installation, make sure to check:
- Add python.exe to PATH
- Install py launcher

Then verify success by running again:

```
py -3.13 --version
```


#### Virtual Environment Setup

First, make sure you are in the project directory:

```
cd path\to\ptm-supplement
```

Now initialize a new virtual environment:

```
py -3.13 -m venv .venv
```

Now activate the virtual environment:

```
.venv\Scripts\activate
```

If successful, your prompt will change and show something like:

```
(.venv) C:\path\to\ptm-supplement>
```


#### Install packages

Now install the required packages:

```
pip install liesel liesel_ptm ipykernel jupyter
```


#### Launch Jupyter Notebook

Now you can launch the demo notebooks:

```
jupyter notebook demos\dutch-growth-study.ipynb
```

```
jupyter notebook demos\framingham-heart-study.ipynb
```