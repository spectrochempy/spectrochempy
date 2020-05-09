# Creating conda environments

In this directory, two yaml file serve for the creation of initial environments:

* scpy.yml is used for a normal user install:

  ```bash
      $ conda deactivate
      $ conda env create -f=env/scpy.yml

    ```
  Modify the ``env/scpy.yml`` string with the full path if you are not in the 
  top-level spectrochempy directory (where you have 
  unzipped the spectrochempy package). 
  
  This will create a ``scpy`` environment, that can be activated using
  
   ```bash
      $ conda activate scpy

    ```
  
* scpy-dev is recommended for a developper: 
  
  ```bash
      $ conda deactivate
      $ conda env create -f=env/scpy-dev.yml

    ```
    
  This will create a ``scpy`` environment, that can be activated using
  
   ```bash
      $ conda activate scpy-dev

    ```
    
To check that the environments have been sussefully created, use:

```bash
$ conda env list
```

This shoud return something like:

```
# conda environments:
#
base                     /Users/username/miniconda3
scpy                     /Users/username/miniconda3/envs/scpy
scpy-dev             *   /Users/username/miniconda3/envs/scpy-dev
```

The star showing which environment is active.

## Jupyter lab 

To be able to use matplotlib in jupyter lab`` we need to run these two command`:

```bash
$ jupyter labextension install @jupyter-widgets/jupyterlab-manager
$ jupyter labextension install jupyter-matplotlib

```