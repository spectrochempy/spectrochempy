The debugger front-end can be installed as a JupyterLab extension.
jupyter labextension install @jupyterlab/debugger

In the back-end, a kernel implementing the Jupyter Debug Protocol is required.
conda install xeus-python -c conda-forge
