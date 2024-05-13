




# Getting Started: 
The first step is to identify a folder location where you would like to work in a development environment.
We suggest a location that will be able to easily access streamflow predictions to make for easy evaluation of your model.
Using the command prompt, change your working directory to this folder and git clone [NWM-ML](https://github.com/whitelightning450/NWM-ML)

    git clone https://github.com/whitelightning450/NWM-ML


# Virtual Environment
It is a best practice to create a virtual environment when starting a new project, as a virtual environment essentially creates an isolated working copy of Python for a particular project. 
I.e., each environment can have its own dependencies or even its own Python versions.
Creating a Python virtual environment is useful if you need different versions of Python or packages for different projects.
Lastly, a virtual environment keeps things tidy, makes sure your main Python installation stays healthy and supports reproducible and open science.

## Creating Stable CONDA Environment on HPC platforms
Go to home directory
```
cd ~
```
Create a envs directory
```
mkdir envs
```
Create .condarc file and link it to a text file
```
touch .condarc

ln -s .condarc condarc.txt
```
Add the below lines to the condarc.txt file
```
# .condarc
envs_dirs:
 - ~/envs
```
Restart your server

### Creating your Virtual Environment
Since we will be using Jupyter Notebooks for this exercise, we will use the Anaconda command prompt to create our virtual environment. 
We suggest using Mamba rather than conda for installs, conda may be used but will take longer.
In the command line type: 

    mamba env create -f PyTorch_GPU_env.yaml 

After a successful environment creation:

    conda activate PyTorch_GPU_env 

You should now be working in your new PyTorch_GPU_env within the command prompt. 
However, we will want to work in this environment within our Jupyter Notebook and need to create a kernel to connect them.
We begin by installing the **ipykernel** python package:

    pip install --user ipykernel

With the package installed, we can connect the PyTorch_GPU_env to our Python Notebook

    python -m ipykernel install --user --name=PyTorch_GPU_env 

The PyTorch_GPU_env should show up on the top right of the Jupyter Notebook.

### Connect to AWS
All of the data for the project is on a publicly accessible AWS S3 bucket (national-snow-model), however, some methods require credentials. 
Please request credentials as an issue and put the credentials in the head of the repo (e.g., SWEMLv2.0) as AWSaccessKeys.csv.

