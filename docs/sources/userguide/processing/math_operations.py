# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.10.8
# ---

# %% [markdown]
# # Mathematical operations

# %%
import numpy as np

import spectrochempy as scp
from spectrochempy import MASKED
from spectrochempy import DimensionalityError
from spectrochempy import error_

# %% [markdown]
# ## Ufuncs (Universal Numpy's functions)
# A universal function (or `ufunc` in short) is a function that operates on numpy arrays in an element-by-element
# fashion, supporting array broadcasting, type casting, and several other standard features. That is, a `ufunc` is a
# “vectorized” wrapper for a function that takes a fixed number of specific inputs and produces a fixed number of
# specific outputs.
#
# For instance, in numpy to calculate the square root of each element of a given nd-array, we can write something
# like this using the `np.sqrt` functions :

# %%
x = np.array([1.0, 2.0, 3.0, 4.0, 6.0])
np.sqrt(x)

# %% [markdown]
# As seen above, `np.sqrt(x)` return a numpy array.
#
# The interesting thing, it that `ufunc`'s can also work with `NDDataset` .

# %%
dx = scp.NDDataset(x)
np.sqrt(dx)

# %% [markdown]
# ## List of UFuncs working on `NDDataset`:
#
# ### Functions affecting magnitudes of the number but keeping units
# * [negative](#negative)(x, **kwargs): Numerical negative, element-wise.
# * [absolute](#abs)(x, **kwargs): Calculate the absolute value, element-wise. Alias: [abs](#abs)
# * [fabs](#abs)(x, **kwargs): Calculate the absolute value, element-wise. Complex values are not handled,
# use [absolute](#abs) to find the absolute values of complex data.
# * [conj](#)(x, **kwargs): Return the complex conjugate, element-wise.
# * [rint](#rint)(x, **kwargs) :Round to the nearest integer, element-wise.
# * [floor](#floor)(x, **kwargs): Return the floor of the input, element-wise.
# * [ceil](#ceil)(x, **kwargs): Return the ceiling of the input, element-wise.
# * [trunc](#trunc)(x, **kwargs): Return the truncated value of the input, element-wise.
#
# ### Functions affecting magnitudes of the number but also units
# * [sqrt](#sqrt)(x, **kwargs): Return the non-negative square-root of an array, element-wise.
# * [square](#square)(x, **kwargs): Return the element-wise square of the input.
# * [cbrt](#cbrt)(x, **kwargs): Return the cube-root of an array, element-wise.
# * [reciprocal](#reciprocal)(x, **kwargs): Return the reciprocal of the argument, element-wise.
#
# ### Functions that require no units or dimensionless units for inputs. Returns dimensionless objects.
# * [exp](#exp)(x, **kwargs): Calculate the exponential of all elements in the input array.
# * [exp2](#exp)(x, **kwargs): Calculate 2**p for all p in the input array.
# * [expm1](#exp)(x, **kwargs): Calculate `exp(x) - 1` for all elements in the array.
# * [log](#log)(x, **kwargs): Natural logarithm, element-wise.
# * [log2](#log)(x, **kwargs): Base-2 logarithm of x.
# * [log10](#log)(x, **kwargs): Return the base 10 logarithm of the input array, element-wise.
# * [log1p](#log)(x, **kwargs): Return `log(x + 1)` , element-wise.
#
# ### Functions that return numpy arrays (*Work only for NDDataset*)
# * [sign](#sign)(x): Returns an element-wise indication of the sign of a number.
# * [logical_not](#logical_not)(x): Compute the truth value of NOT x element-wise.
# * [isfinite](#isfinite)(x): Test element-wise for finiteness.
# * [isinf](#isinf)(x): Test element-wise for positive or negative infinity.
# * [isnan](#isnan)(x): Test element-wise for `NaN` and return result as a boolean array.
# * [signbit](#signbit)(x): Returns element-wise `True` where signbit is set.
#
# ### Trigonometric functions. Require unitless data or radian units.
# * [sin](#sin)(x, **kwargs): Trigonometric sine, element-wise.
# * [cos](#cos)(x, **kwargs): Trigonometric cosine element-wise.
# * [tan](#tan)(x, **kwargs): Compute tangent element-wise.
# * [arcsin](#arcsin)(x, **kwargs): Inverse sine, element-wise.
# * [arccos](#arccos)(x, **kwargs): Trigonometric inverse cosine, element-wise.
# * [arctan](#arctan)(x, **kwargs): Trigonometric inverse tangent, element-wise.
#
# ### Hyperbolic functions
# * [sinh](#sinh)(x, **kwargs): Hyperbolic sine, element-wise.
# * [cosh](#cosh)(x, **kwargs): Hyperbolic cosine, element-wise.
# * [tanh](#tanh)(x, **kwargs): Compute hyperbolic tangent element-wise.
# * [arcsinh](#arcsinh)(x, **kwargs): Inverse hyperbolic sine element-wise.
# * [arccosh](#arccosh)(x, **kwargs): Inverse hyperbolic cosine, element-wise.
# * [arctanh](#arctanh)(x, **kwargs): Inverse hyperbolic tangent element-wise.
#
# ### Unit conversions
# * [deg2rad](#deg2rad)(x, **kwargs): Convert angles from degrees to radians.
# * [rad2deg](#rad2deg)(x, **kwargs): Convert angles from radians to degrees.
#
# ### Binary Ufuncs
#
# * [add](#add)(x1, x2, **kwargs): Add arguments element-wise.
# * [subtract](#subtract)(x1, x2, **kwargs): Subtract arguments, element-wise.
# * [multiply](#multiply)(x1, x2, **kwargs): Multiply arguments element-wise.
# * [divide](#divide) or [true_divide](#divide)(x1, x2, **kwargs): Returns a true division of the inputs,
# element-wise.
# * [floor_divide](#floor_divide)(x1, x2, **kwargs): Return the largest integer smaller or equal to the division of
# the inputs.

# %% [markdown]
# ## Usage
# To demonstrate the use of mathematical operations on spectrochempy object, we will first load an experimental 2D
# dataset.

# %%
d2D = scp.read_omnic("irdata/nh4y-activation.spg")
prefs = d2D.preferences
prefs.colormap = "magma"
prefs.colorbar = False
prefs.figure.figsize = (6, 3)
d2D.plot()

# %% [markdown]
# Let's select only the first row of the 2D dataset ( the `squeeze` method is used to remove
# the residual size 1 dimension). In addition, we mask the saturated region.

# %%
dataset = d2D[0].squeeze()
dataset.plot()

# %% [markdown]
# This dataset will be artificially modified already using some mathematical operation (subtraction with a scalar) to
# present negative values, and we will also mask some data

# %%
dataset -= 2.0  # add an offset to make that some of the values become negative
dataset[1290.0:890.0] = scp.MASKED  # additionally we mask some data
dataset.plot()

# %% [markdown]
# ### Unary functions

# %% [markdown]
# #### Functions affecting magnitudes of the number but keeping units

# %% [markdown]
# ##### negative
# Numerical negative, element-wise, keep units

# %%
out = np.negative(dataset)  # the same results is obtained using out=-dataset
out.plot(figsize=(6, 2.5), show_mask=True)

# %% [markdown]
# ##### abs
# ##### absolute (alias of abs)
# ##### fabs (absolute for float arrays)
# Numerical absolute value element-wise, element-wise, keep units

# %%
out = np.abs(dataset)
out.plot(figsize=(6, 2.5))

# %% [markdown]
# ##### rint
# Round elements of the array to the nearest integer, element-wise, keep units

# %%
out = np.rint(dataset)
out.plot(figsize=(6, 2.5))  # not that title is not modified for this ufunc

# %% [markdown]
# ##### floor
# Return the floor of the input, element-wise.

# %%
out = np.floor(dataset)
out.plot(figsize=(6, 2.5))

# %% [markdown]
# ##### ceil
# Return the ceiling of the input, element-wise.

# %%
out = np.ceil(dataset)
out.plot(figsize=(6, 2.5))

# %% [markdown]
# ##### trunc
# Return the truncated value of the input, element-wise.

# %%
out = np.trunc(dataset)
out.plot(figsize=(6, 2.5))

# %% [markdown]
# #### Functions affecting magnitudes of the number but also units
# ##### sqrt
# Return the non-negative square-root of an array, element-wise.

# %%
out = np.sqrt(
    dataset
)  # as they are some negative elements, return dataset has complex dtype.
out.plot_1D(show_complex=True, figsize=(6, 2.5))

# %% [markdown]
# ##### square
# Return the element-wise square of the input.

# %%
out = np.square(dataset)
out.plot(figsize=(6, 2.5))

# %% [markdown]
# ##### cbrt
# Return the cube-root of an array, element-wise.

# %%
out = np.cbrt(dataset)
out.plot(figsize=(6, 2.5))

# %% [markdown]
# ##### reciprocal
# Return the reciprocal of the argument, element-wise.

# %%
out = np.reciprocal(dataset + 3.0)
out.plot(figsize=(6, 2.5))

# %% [markdown]
# #### Functions that require no units or dimensionless units for inputs. Returns dimensionless objects.

# %% [markdown]
# ##### exp
# Exponential of all elements in the input array, element-wise

# %%
out = np.exp(dataset)
out.plot(figsize=(6, 2.5))

# %% [markdown]
# Obviously numpy exponential functions applies only to dimensionless array. Else an error is generated.

# %%
x = scp.NDDataset(np.arange(5), units="m")
try:
    np.exp(x)  # A dimensionality error will be generated
except DimensionalityError as e:
    error_(DimensionalityError, e)

# %% [markdown]
# ##### exp2
# Calculate 2**p for all p in the input array.

# %%
out = np.exp2(dataset)
out.plot(figsize=(6, 2.5))

# %% [markdown]
# ##### expm1
# Calculate `exp(x) - 1` for all elements in the array.

# %%
out = np.expm1(dataset)
out.plot(figsize=(6, 2.5))

# %% [markdown]
# ##### log
# Natural logarithm, element-wise.
#
# This doesn't generate un error for negative numbrs, but the output is masked for those values

# %%
out = np.log(dataset)
ax = out.plot(figsize=(6, 2.5), show_mask=True)

# %%
out = np.log(dataset - dataset.min())
out.plot(figsize=(6, 2.5))

# %% [markdown]
# ##### log2
# Base-2 logarithm of x.

# %%
out = np.log2(dataset)
out.plot(figsize=(6, 2.5))

# %% [markdown]
# ##### log10
# Return the base 10 logarithm of the input array, element-wise.

# %%
out = np.log10(dataset)
out.plot(figsize=(6, 2.5))

# %% [markdown]
# ##### log1p
# Return `log(x + 1)` , element-wise.

# %%
out = np.log1p(dataset)
out.plot(figsize=(6, 2.5))

# %% [markdown]
# #### Functions that return numpy arrays (*Work only for NDDataset*)

# %% [markdown]
# ##### sign
# Returns an element-wise indication of the sign of a number. Returned object is a ndarray

# %%
np.sign(dataset)

# %% [markdown]
# ##### logical_not
# Compute the truth value of NOT x element-wise. Returned object is a ndarray
# %%
np.logical_not(dataset < 0)

# %% [markdown]
# ##### isfinite
# Test element-wise for finiteness.

# %%
np.isfinite(dataset)

# %% [markdown]
# ##### isinf
# Test element-wise for positive or negative infinity.

# %%
np.isinf(dataset)

# %% [markdown]
# ##### isnan
# Test element-wise for `NaN` and return result as a boolean array.

# %%
np.isnan(dataset)

# %% [markdown]
# ##### signbit
# Returns element-wise `True` where signbit is set.

# %%
np.signbit(dataset)

# %% [markdown]
# #### Trigonometric functions. Require dimensionless/unitless  dataset or radians.
#
# In the below examples, unit of data in dataset is absorbance (then dimensionless)

# %% [markdown]
# ##### sin
# Trigonometric sine, element-wise.

# %%
out = np.sin(dataset)
out.plot(figsize=(6, 2.5))

# %% [markdown]
# ##### cos
# Trigonometric cosine element-wise.

# %%
out = np.cos(dataset)
out.plot(figsize=(6, 2.5))

# %% [markdown]
# ##### tan
# Compute tangent element-wise.

# %%
out = np.tan(dataset / np.max(dataset))
out.plot(figsize=(6, 2.5))

# %% [markdown]
# ##### arcsin
# Inverse sine, element-wise.

# %%
out = np.arcsin(dataset)
out.plot(figsize=(6, 2.5))

# %% [markdown]
# ##### arccos
# Trigonometric inverse cosine, element-wise.

# %%
out = np.arccos(dataset)
out.plot(figsize=(6, 2.5))

# %% [markdown]
# ##### arctan
# Trigonometric inverse tangent, element-wise.

# %%
out = np.arctan(dataset)
out.plot(figsize=(6, 2.5))

# %% [markdown]
# #### Angle units conversion

# %% [markdown]
# ##### rad2deg
# Convert angles from radians to degrees (warning: unitless or dimensionless are assumed to be radians, so no error
# will be issued).


# %% [markdown]
# for instance, if we take the z axis (the data magnitude) in the figure above, it's expressed in radians. We can
# change to degrees easily.


# %%
out = np.rad2deg(dataset)
out.title = "data"  # just to avoid a too long title
out.plot(figsize=(6, 2.5))

# %% [markdown]
# ##### deg2rad
# Convert angles from degrees to radians.


# %%
out = np.deg2rad(out)
out.title = "data"
out.plot(figsize=(6, 2.5))

# %% [markdown]
# #### Hyperbolic functions

# %% [markdown]
# ##### sinh
# Hyperbolic sine, element-wise.

# %%
out = np.sinh(dataset)
out.plot(figsize=(6, 2.5))

# %% [markdown]
# ##### cosh
# Hyperbolic cosine, element-wise.

# %%
out = np.cosh(dataset)
out.plot(figsize=(6, 2.5))

# %% [markdown]
# ##### tanh
# Compute hyperbolic tangent element-wise.

# %%
out = np.tanh(dataset)
out.plot(figsize=(6, 2.5))

# %% [markdown]
# ##### arcsinh
# Inverse hyperbolic sine element-wise.

# %%
out = np.arcsinh(dataset)
out.plot(figsize=(6, 2.5))

# %% [markdown]
# ##### arccosh
# Inverse hyperbolic cosine, element-wise.

# %%
out = np.arccosh(dataset)
out.plot(figsize=(6, 2.5))

# %% [markdown]
# ##### arctanh
# Inverse hyperbolic tangent element-wise.

# %%
out = np.arctanh(dataset)
out.plot(figsize=(6, 2.5))

# %% [markdown]
# ### Binary functions

# %%
dataset2 = np.reciprocal(dataset + 3)  # create a second dataset
dataset2[5000.0:4000.0] = MASKED
dataset.plot(figsize=(6, 2.5))
dataset2.plot(figsize=(6, 2.5))

# %% [markdown]
# #### Arithmetic

# %% [markdown]
# ##### add
# Add arguments element-wise.

# %%
out = np.add(dataset, dataset2)
out.plot(figsize=(6, 2.5))

# %% [markdown]
# ##### subtract
# Subtract arguments, element-wise.

# %%
out = np.subtract(dataset, dataset2)
out.plot(figsize=(6, 2.5))

# %% [markdown]
# ##### multiply
# Multiply arguments element-wise.

# %%
out = np.multiply(dataset, dataset2)
out.plot(figsize=(6, 2.5))

# %% [markdown]
# ##### divide
# or
# ##### true_divide
# Returns a true division of the inputs, element-wise.

# %%
out = np.divide(dataset, dataset2)
out.plot(figsize=(6, 2.5))

# %% [markdown]
# ##### floor_divide
# Return the largest integer smaller or equal to the division of the inputs.

# %%
out = np.floor_divide(dataset, dataset2)
out.plot(figsize=(6, 2.5))

# %% [markdown]
# ## Complex or hypercomplex NDDatasets
#
#
# NDDataset objects with complex data are handled differently than in
# `numpy.ndarray` .
#
# Instead, complex data are stored by interlacing the real and imaginary part.
# This allows the definition of data that can be complex in several axis, and *e
# .g.,* allows 2D-hypercomplex array that can be transposed (useful for NMR data).

# %%
da = scp.NDDataset(
    [
        [1.0 + 2.0j, 2.0 + 0j],
        [1.3 + 2.0j, 2.0 + 0.5j],
        [1.0 + 4.2j, 2.0 + 3j],
        [5.0 + 4.2j, 2.0 + 3j],
    ]
)
da

# %% [markdown]
# A dataset of type float can be transformed into a complex dataset (using two consecutive rows to create a complex
# row)

# %%
da = scp.NDDataset(np.arange(40).reshape(10, 4))
da

# %%
dac = da.set_complex()
dac

# %% [markdown]
# Note the `x`dimension size is divided by a factor of two

# %% [markdown]
# A dataset which is complex in two dimensions is called hypercomplex (it's datatype in SpectroChemPy is set to
# quaternion).

# %%
daq = da.set_quaternion()  # equivalently one can use the set_hypercomplex method
daq

# %%
daq.dtype
