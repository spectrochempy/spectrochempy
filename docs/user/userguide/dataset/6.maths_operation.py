# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Mathematical operations

# %%
from spectrochempy import *

# %% [markdown]
# ## Ufuncs (Universal Numpy's functions)
# A universal function (or `ufunc` in short) is a function that operates on numpy arrays in an element-by-element fashion, supporting array broadcasting, type casting, and several other standard features. That is, a `ufunc` is a “vectorized” wrapper for a function that takes a fixed number of specific inputs and produces a fixed number of specific outputs.
#
# For instance, in numpy to calculate the square root of each element of a given nd-array, we can write something like this using the `np.sqrt` functions :

# %%
x = np.array([1.,2.,3.,4.,6.])
np.sqrt(x)

# %% [markdown]
# As seen above, `np.sqrt(x)` return a numpy array.
#
# The interesting thing, it that `ufunc`'s can also work with `NDDataset`.

# %%
dx = NDDataset(x)
np.sqrt(dx)

# %% [markdown]
# In this case a `NDDataset` object is returned.
#
# A subset of `ufunc`'s also work works with `NDPanel`. Ufunc is then broadcasted to all internal arrays.

# %%
x = np.arange(6.)
y = np.arange(8.)
pnl = NDPanel(x, y)
np.sqrt(pnl)

# %% [markdown]
# As observed in the example above, in this case a `NDPanel`is returned on which each internal datasets have been transformed by the `ufunc`.

# %% [markdown]
# It is worth to note, that in spectrochempy, when can use internal equivalent of these `ufunc`'s.
#
# For instance, the square root can be calculated using the following syntax:

# %%
import spectrochempy as scp 
scp.sqrt(dx)

# %% [markdown]
# or

# %%
from spectrochempy import *
sqrt(dx)

# %% [markdown]
# ## List of UFuncs working on `NDDataset`:
#
# if an Ufunc works on `NDDataset` but not on `NDPanel`, this is specified in the list below.
#
# ### Functions affecting magnitudes of the number but keeping units
# * [negative](#negative)(x, \*\*kwargs): Numerical negative, element-wise. 
# * [absolute](#abs)(x, \*\*kwargs): Calculate the absolute value, element-wise. Alias: [abs](#abs)
# * [fabs](#abs)(x, \*\*kwargs): Calculate the absolute value, element-wise. Complex values are not handled, use [absolute](#absolute) to find the absolute values of complex data.
# * [conj](#)(x, \*\*kwargs): Return the complex conjugate, element-wise. 
# * [rint](#rint)(x, \*\*kwargs) :Round to the nearest integer, element-wise.  
# * [floor](#floor)(x, \*\*kwargs): Return the floor of the input, element-wise.
# * [ceil](#ceil)(x, \*\*kwargs): Return the ceiling of the input, element-wise.
# * [trunc](#trunc)(x, \*\*kwargs): Return the truncated value of the input, element-wise. 
#
# ### Functions affecting magnitudes of the number but also units
# * [sqrt](#sqrt)(x, \*\*kwargs): Return the non-negative square-root of an array, element-wise.
# * [square](#square)(x, \*\*kwargs): Return the element-wise square of the input.
# * [cbrt](#cbrt)(x, \*\*kwargs): Return the cube-root of an array, element-wise.
# * [reciprocal](#reciprocal)(x, \*\*kwargs): Return the reciprocal of the argument, element-wise.
#
# ### Functions that require no units or dimensionless units for inputs. Returns dimensionless objects.
# * [exp](#exp)(x, \*\*kwargs): Calculate the exponential of all elements in the input array.
# * [exp2](#exp)(x, \*\*kwargs): Calculate 2\*\*p for all p in the input array.
# * [expm1](#exp)(x, \*\*kwargs): Calculate `exp(x) - 1` for all elements in the array. 
# * [log](#log)(x, \*\*kwargs): Natural logarithm, element-wise. 
# * [log2](#log)(x, \*\*kwargs): Base-2 logarithm of x.
# * [log10](#log)(x, \*\*kwargs): Return the base 10 logarithm of the input array, element-wise. 
# * [log1p](#log)(x, \*\*kwargs): Return `log(x + 1)`, element-wise.
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
# * [sin](#sin)(x, \*\*kwargs): Trigonometric sine, element-wise.
# * [cos](#cos)(x, \*\*kwargs): Trigonometric cosine element-wise.
# * [tan](#tan)(x, \*\*kwargs): Compute tangent element-wise. 
#
# * [arcsin](#arcsin)(x, \*\*kwargs): Inverse sine, element-wise.
# * [arccos](#arccos)(x, \*\*kwargs): Trigonometric inverse cosine, element-wise.
# * [arctan](#arctan)(x, \*\*kwargs): Trigonometric inverse tangent, element-wise.
#
# ### Hyperbolic functions
# * [sinh](#sinh)(x, \*\*kwargs): Hyperbolic sine, element-wise.
# * [cosh](#cosh)(x, \*\*kwargs): Hyperbolic cosine, element-wise.
# * [tanh](#tanh)(x, \*\*kwargs): Compute hyperbolic tangent element-wise.
# * [arcsinh](#arcsinh)(x, \*\*kwargs): Inverse hyperbolic sine element-wise.
# * [arccosh](#arccosh)(x, \*\*kwargs): Inverse hyperbolic cosine, element-wise.
# * [arctanh](#arctanh)(x, \*\*kwargs): Inverse hyperbolic tangent element-wise.
#
# ### Unit conversions
# * [deg2rad](#deg2rad)(x, \*\*kwargs): Convert angles from degrees to radians.
# * [rad2deg](#rad2deg)(x, \*\*kwargs): Convert angles from radians to degrees.
#
# ### Binary Ufuncs
#
# * [add](#add)(x1, x2, \*\*kwargs): Add arguments élement-wise.
# * [subtract](#subtract)(x1, x2, \*\*kwargs): Subtract arguments, element-wise.
# * [multiply](#multiply)(x1, x2, \*\*kwargs): Multiply arguments element-wise.
# * [divide](#divide) or [true_divide](#true_divide)(x1, x2, \*\*kwargs): Returns a true division of the inputs, element-wise.
# * [floor_divide](#floor_divide)(x1, x2, \*\*kwargs): Return the largest integer smaller or equal to the division of the inputs.
# * [mod](#mod) or [remainder](#remainder)(x1, x2,\*\*kwargs): Return element-wise remainder of division.
# * [fmod](#fmod)(x1, x2, \*\*kwargs): Return the element-wise remainder of division. 
# * [logaddexp](#logaddexp)(x1, x2,\*\*kwargs): Logarithm of the sum of exponentiations of the inputs.
# * [logaddexp2](#logaddexp2)(x1, x2, \*\*kwargs): Logarithm of the sum of exponentiations of the inputs in base-2.
# * [heaviside](#heaviside)(x1, x2, \*\*kwargs): Compute the Heaviside step function.
#
#

# %% [markdown]
# ## Usage
# To demonstrate the use of mathematical operations on spectrochempy object, we will first load an experimental 2D dataset.

# %%
# import a dataset to check the ufuncs
dataset = NDDataset.read_omnic(os.path.join('irdata', 'nh4y-activation.spg'))
dataset

# %% [markdown]
# This dataset will be artificially modified already using some mathematical operation (subtraction with a scalar) to present negative values.

# %%
dataset = dataset - 2.     # add an offset to make that some of the values become negative
dataset

# %%
dataset = dataset.clip(1.8)  # additionally limit the upper values using the clip 
dataset

# %%
_ = dataset.plot(figsize=(6,2.5))

# %% [markdown]
# ### Unary functions

# %% [markdown]
# #### Functions affecting magnitudes of the number but keeping units

# %% [markdown]
# ##### negative
# Numerical negative, element-wise, keep units

# %%
out = np.negative(dataset)
_ = out.plot(figsize=(6,2.5))

# %% [markdown]
# ##### abs
# ##### absolute (alias of abs)
# ##### fabs (absolute for float arrays)
# Numerical absolute value element-wise, element-wise, keep units

# %%
out = np.abs(dataset)
_ = out.plot(figsize=(6,2.5))

# %% [markdown]
# ##### rint
# Round elements of the array to the nearest integer, element-wise, keep units

# %%
out = np.rint(dataset)
_ = out.plot(figsize=(6,2.5))  # not that title is not modified for this ufunc

# %% [markdown]
# ##### floor
# Return the floor of the input, element-wise.

# %%
out = np.floor(dataset)
_ = out.plot(figsize=(6,2.5)) 

# %% [markdown]
# ##### ceil
# Return the ceiling of the input, element-wise.

# %%
out = np.ceil(dataset)
_ = out.plot(figsize=(6,2.5)) 

# %% [markdown]
# ##### trunc
# Return the truncated value of the input, element-wise.

# %%
out = np.trunc(dataset)
_ = out.plot(figsize=(6,2.5)) 

# %% [markdown]
# #### Functions affecting magnitudes of the number but also units
# ##### sqrt
# Return the non-negative square-root of an array, element-wise.

# %%
out = np.sqrt(dataset)   # as they are some negative elements, return dataset has complex dtype.
_ = out.real.plot(figsize=(6,2.5))
_ = out.imag.plot(figsize=(6,2.5))

# %% [markdown]
# ##### square
# Return the element-wise square of the input.

# %%
out = np.square(dataset)
_ = out.plot(figsize=(6,2.5))

# %% [markdown]
# ##### cbrt
# Return the cube-root of an array, element-wise.

# %%
out = np.cbrt(dataset)
_ = out.plot(figsize=(6,2.5))

# %% [markdown]
# ##### reciprocal
# Return the reciprocal of the argument, element-wise.

# %%
out = np.reciprocal(dataset+3.)
_ = out.plot(figsize=(6,2.5))

# %% [markdown]
# #### Functions that require no units or dimensionless units for inputs. Returns dimensionless objects.

# %% [markdown]
# ##### exp
# Exponential of all elements in the input array, element-wise

# %%
out = np.exp(dataset)
_ = out.plot(figsize=(6,2.5))

# %% [markdown]
# Obviously numpy exponential functions applies only to dimensionless array. Else an error is generated.

# %%
x = NDDataset(np.arange(5), units='m')
try:
    np.exp(x)  # A dimensionality error will be generated
except DimensionalityError as e:
    #error_(e)
    print('Dimensionality error issued!')

# %% [markdown]
# ##### exp2
# Calculate 2\*\*p for all p in the input array.

# %%
out = np.exp2(dataset)
_ = out.plot(figsize=(6,2.5))

# %% [markdown]
# ##### expm1 
# Calculate `exp(x) - 1` for all elements in the array. 

# %%
out = np.expm1(dataset)
_ = out.plot(figsize=(6,2.5))

# %% [markdown]
# ##### log
# Natural logarithm, element-wise. 

# %%
out = np.log(dataset)
_ = out.plot(figsize=(6,2.5))

# %% [markdown]
# ##### log2
# Base-2 logarithm of x.

# %%
out = np.log2(dataset)
_ = out.plot(figsize=(6,2.5))

# %% [markdown]
# ##### log10
# Return the base 10 logarithm of the input array, element-wise. 

# %%
out = np.log10(dataset)
_ = out.plot(figsize=(6,2.5))

# %% [markdown]
# ##### log1p
# Return `log(x + 1)`, element-wise.

# %%
out = np.log1p(dataset)
_ = out.plot(figsize=(6,2.5))

# %% [markdown]
# #### Functions that return numpy arrays (*Work only for NDDataset*)

# %% [markdown]
# ##### sign
# Returns an element-wise indication of the sign of a number. Returned object is a ndarray

# %%
np.sign(dataset)

# %%
np.logical_not(dataset<0)

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
# #### Trigonometric functions. Require unitless dataset or radians.

# %% [markdown]
# ##### sin
# Trigonometric sine, element-wise.

# %%
out = np.sin(dataset)
_ = out.plot(figsize=(6,2.5))

# %% [markdown]
# ##### cos
# Trigonometric cosine element-wise.

# %%
out = np.cos(dataset)
_ = out.plot(figsize=(6,2.5))

# %% [markdown]
# ##### tan
# Compute tangent element-wise.

# %%
out = np.tan(dataset/np.max(dataset))
_ = out.plot(figsize=(6,2.5))

# %% [markdown]
# ##### arcsin
# Inverse sine, element-wise.

# %%
out = np.arcsin(dataset)
_ = out.plot(figsize=(6,2.5))

# %% [markdown]
# ##### arccos
# Trigonometric inverse cosine, element-wise.

# %%
out = np.arccos(dataset)
_ = out.plot(figsize=(6,2.5))

# %% [markdown]
# ##### arctan
# Trigonometric inverse tangent, element-wise.

# %%
out = np.arctan(dataset)
_ = out.plot(figsize=(6,2.5))


# %% [markdown]
# ##### arctan2
# Element-wise arc tangent of x1/x2 choosing the quadrant correctly.
#
# The quadrant (i.e., branch) is chosen so that arctan2(x1, x2) is the signed angle in radians between the ray ending at the origin and passing through the point (1,0), and the ray ending at the origin and passing through the point (x2, x1). 

# %%
out = np.arctan2(dataset, dataset*2.)
_ = out.plot(figsize=(6,2.5))

# %% [markdown]
# #### Angle units conversion

# %% [markdown]
# ##### rad2deg
# Convert angles from radians to degrees (warning: unitless or dimensionless are assumed to be radians, so no error will be issued).


# %% [markdown]
# for instance, if we take the z axis (the data magintude) in the figure above, it's expressed in radians. We can change to degrees easily.


# %%
out = np.rad2deg(dataset)
out.title = 'data' # just to avoid a too long title
_ = out.plot(figsize=(6,2.5))

# %%
pi = NDDataset(Quantity(np.pi,'dimensionless'))
np.sin(pi).data, np.sin(np.rad2deg(pi)).data

# %% [markdown]
# ##### deg2rad
# Convert angles from degrees to radians.


# %%
out = np.deg2rad(out)
out.title = 'data' 
_ = out.plot(figsize=(6,2.5))

# %% [markdown]
# #### Hyperbolic functions

# %% [markdown]
# ##### sinh
# Hyperbolic sine, element-wise.

# %%
out = np.sinh(dataset)
_ = out.plot(figsize=(6,2.5))


# %% [markdown]
# ##### cosh
# Hyperbolic cosine, element-wise.

# %%
out = np.cosh(dataset)
_ = out.plot(figsize=(6,2.5))


# %% [markdown]
# ##### tanh
# Compute hyperbolic tangent element-wise.

# %%
out = np.tanh(dataset)
_ = out.plot(figsize=(6,2.5))


# %% [markdown]
# ##### arcsinh
# Inverse hyperbolic sine element-wise.

# %%
out = np.arcsinh(dataset)
_ = out.plot(figsize=(6,2.5))


# %% [markdown]
# ##### arccosh
# Inverse hyperbolic cosine, element-wise.

# %% [markdown]
# ##### arctanh
# Inverse hyperbolic tangent element-wise.

# %%
out = np.arctanh(dataset)
_ = out.plot(figsize=(6,2.5))


# %% [markdown]
# ### Binary functions

# %%
dataset2 = np.exp(dataset)  # create a second dataset
_ = dataset.plot(figsize=(6,2.5))
_ = dataset2.plot(figsize=(6,2.5))

# %% [markdown]
# #### Arithmetics

# %% [markdown]
# ##### add
# Add arguments element-wise.

# %%
out = np.add(dataset, dataset2)
_ = out.plot(figsize=(6,2.5))


# %% [markdown]
# ##### subtract
# Subtract arguments, element-wise.

# %%
out = np.subtract(dataset, dataset2)
_ = out.plot(figsize=(6,2.5))


# %% [markdown]
# ##### multiply
# Multiply arguments element-wise.

# %%
out = np.multiply(dataset, dataset2)
_ = out.plot(figsize=(6,2.5))


# %% [markdown]
# ##### divide
# or
# ##### true_divide
# Returns a true division of the inputs, element-wise.

# %%
out = np.divide(dataset, dataset2)
_ = out.plot(figsize=(6,2.5))


# %% [markdown]
# ##### floor_divide
# Return the largest integer smaller or equal to the division of the inputs.

# %%
out = np.floor_divide(dataset, dataset2)
_ = out.plot(figsize=(6,2.5))


# %% [markdown]
# ##### mod
# or
# ##### remainder
# Return element-wise remainder of division.

# %%
out = np.mod(dataset, dataset2)
_ = out.plot(figsize=(6,2.5))


# %% [markdown]
# ##### fmod
# Return element-wise remainder of division.
#
# **Note**: For `fmod`, the sign of result is the sign of the dividend, while for remainder the sign of the result is the sign of the divisor.

# %%
out = np.fmod(dataset, dataset2)
_ = out.plot(figsize=(6,2.5))


# %% [markdown]
# ##### logaddexp
# Logarithm of the sum of exponentiations of the inputs.
# ##### logaddexp2
# Logarithm of the sum of exponentiations of the inputs in base-2.

# %%
out = np.logaddexp2(dataset, dataset2)
_ = out.plot(figsize=(6,2.5))


# %% [markdown]
# ##### heaviside
# Compute the Heaviside step function.
#
# ```
#                       0   if x1 < 0
# heaviside(x1, x2) =  x2   if x1 == 0
#                       1   if x1 > 0
# ```

# %%
h = np.heaviside(dataset, dataset2)
_ = h.plot(figsize=(6,2.5))
out = dataset * h
_ = out.plot(figsize=(6,2.5))


# %% [markdown]
# #### Logical and comparison operations

# %% [markdown]
# ##### logical_not
# Compute the truth value of NOT x element-wise. 

# %%
np.logical_not(dataset)
