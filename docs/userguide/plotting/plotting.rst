.. _userguide.plotting

Plotting
#########

`%matplotlib qt`  or `%matplotlib widget`
-----------------------------------------

In `Jupyter Lab` (or `Jupyter Notebook`) the use of the **"magic command "** `%matplotlib widget` triggers the interactive drawing. This process is integrated in the Notebook with basic tools to navigate inside the figure. As shown below, from top to bottom of the
sidebar :

- hide/show tools,
- reset the view ("home"),
- previous view ( left arrow ),
- next view ( right arrow ),
- move ("arrow cross"),
- zoom ( rectangle ),
- save the image ("disk").

<img src="figures/widgetsmode.png" alt="widgets mode" width="700" align="center" />

In recent versions of **SpectroChemPy** (i.e. from versions > 0.1.21) this is the default setting and you do not need to run this command .
Another possibility is to use the magic command `%matplotlib qt` instead. In this case the plot is generated in an external window, with some interactive capability.

<div class = "alert warning">
<WARNING: </b> This selection must be made before loading the API, otherwise it will not be taken into account (and the widget <em>%matplotlib</em> will be used by default)
</div>

In both these interactive mode, the current `x` and `y` coordinate values are shown when the mouse pointer is moved inside  the plot limits.

If you prefer to generate static plots (for a laboratory notebook for example), just enter `%matplotlib inline`.
