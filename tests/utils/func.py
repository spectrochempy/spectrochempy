ufunc_str = """

# Math operations

add(x1, x2, /[, out, where, casting, order, …])	Add arguments element-wise.
subtract(x1, x2, /[, out, where, casting, …])	Subtract arguments, element-wise.
multiply(x1, x2, /[, out, where, casting, …])	Multiply arguments element-wise.
divide(x1, x2, /[, out, where, casting, …])	Returns a true division of the inputs, element-wise.
logaddexp(x1, x2, /[, out, where, casting, …])	Logarithm of the sum of exponentiations of the inputs.
logaddexp2(x1, x2, /[, out, where, casting, …])	Logarithm of the sum of exponentiations of the inputs in base-2.
true_divide(x1, x2, /[, out, where, …])	Returns a true division of the inputs, element-wise.
floor_divide(x1, x2, /[, out, where, …])	Return the largest integer smaller or equal to the division of the inputs.
negative(x, /[, out, where, casting, order, …])	Numerical negative, element-wise.
positive(x, /[, out, where, casting, order, …])	Numerical positive, element-wise.
power(x1, x2, /[, out, where, casting, …])	First array elements raised to powers from second array, element-wise.
remainder(x1, x2, /[, out, where, casting, …])	Return element-wise remainder of division.
mod(x1, x2, /[, out, where, casting, order, …])	Return element-wise remainder of division.
fmod(x1, x2, /[, out, where, casting, …])	Return the element-wise remainder of division.
divmod(x1, x2[, out1, out2], / [[, out, …])	Return element-wise quotient and remainder simultaneously.
absolute(x, /[, out, where, casting, order, …])	Calculate the absolute value element-wise.
fabs(x, /[, out, where, casting, order, …])	Compute the absolute values element-wise.
rint(x, /[, out, where, casting, order, …])	Round elements of the array to the nearest integer.
sign(x, /[, out, where, casting, order, …])	Returns an element-wise indication of the sign of a number.
heaviside(x1, x2, /[, out, where, casting, …])	Compute the Heaviside step function.
conj(x, /[, out, where, casting, order, …])	Return the complex conjugate, element-wise.
exp(x, /[, out, where, casting, order, …])	Calculate the exponential of all elements in the input array.
exp2(x, /[, out, where, casting, order, …])	Calculate 2**p for all p in the input array.
log(x, /[, out, where, casting, order, …])	Natural logarithm, element-wise.
log2(x, /[, out, where, casting, order, …])	Base-2 logarithm of x.
log10(x, /[, out, where, casting, order, …])	Return the base 10 logarithm of the input array, element-wise.
expm1(x, /[, out, where, casting, order, …])	Calculate exp(x) - 1 for all elements in the array.
log1p(x, /[, out, where, casting, order, …])	Return the natural logarithm of one plus the input array, element-wise.
sqrt(x, /[, out, where, casting, order, …])	Return the non-negative square-root of an array, element-wise.
square(x, /[, out, where, casting, order, …])	Return the element-wise square of the input.
cbrt(x, /[, out, where, casting, order, …])	Return the cube-root of an array, element-wise.
reciprocal(x, /[, out, where, casting, …])	Return the reciprocal of the argument, element-wise.
gcd(x1, x2, /[, out, where, casting, order, …])	Returns the greatest common divisor of |x1| and |x2|
lcm(x1, x2, /[, out, where, casting, order, …])	Returns the lowest common multiple of |x1| and |x2|

# Trigonometric functions

sin(x, /[, out, where, casting, order, …])	Trigonometric sine, element-wise.
cos(x, /[, out, where, casting, order, …])	Cosine element-wise.
tan(x, /[, out, where, casting, order, …])	Compute tangent element-wise.
arcsin(x, /[, out, where, casting, order, …])	Inverse sine, element-wise.
arccos(x, /[, out, where, casting, order, …])	Trigonometric inverse cosine, element-wise.
arctan(x, /[, out, where, casting, order, …])	Trigonometric inverse tangent, element-wise.
arctan2(x1, x2, /[, out, where, casting, …])	Element-wise arc tangent of x1/x2 choosing the quadrant correctly.
hypot(x1, x2, /[, out, where, casting, …])	Given the “legs” of a right triangle, return its hypotenuse.
sinh(x, /[, out, where, casting, order, …])	Hyperbolic sine, element-wise.
cosh(x, /[, out, where, casting, order, …])	Hyperbolic cosine, element-wise.
tanh(x, /[, out, where, casting, order, …])	Compute hyperbolic tangent element-wise.
arcsinh(x, /[, out, where, casting, order, …])	Inverse hyperbolic sine element-wise.
arccosh(x, /[, out, where, casting, order, …])	Inverse hyperbolic cosine, element-wise.
arctanh(x, /[, out, where, casting, order, …])	Inverse hyperbolic tangent element-wise.
deg2rad(x, /[, out, where, casting, order, …])	Convert angles from degrees to radians.
rad2deg(x, /[, out, where, casting, order, …])	Convert angles from radians to degrees.

# Comparison functions

greater(x1, x2, /[, out, where, casting, …])	Return the truth value of (x1 > x2) element-wise.
greater_equal(x1, x2, /[, out, where, …])	Return the truth value of (x1 >= x2) element-wise.
less(x1, x2, /[, out, where, casting, …])	Return the truth value of (x1 < x2) element-wise.
less_equal(x1, x2, /[, out, where, casting, …])	Return the truth value of (x1 =< x2) element-wise.
not_equal(x1, x2, /[, out, where, casting, …])	Return (x1 != x2) element-wise.
equal(x1, x2, /[, out, where, casting, …])	Return (x1 == x2) element-wise.
logical_and(x1, x2, /[, out, where, …])	Compute the truth value of x1 AND x2 element-wise.
logical_or(x1, x2, /[, out, where, casting, …])	Compute the truth value of x1 OR x2 element-wise.
logical_xor(x1, x2, /[, out, where, …])	Compute the truth value of x1 XOR x2, element-wise.
logical_not(x, /[, out, where, casting, …])	Compute the truth value of NOT x element-wise.
maximum(x1, x2, /[, out, where, casting, …])	Element-wise maximum of array elements.
minimum(x1, x2, /[, out, where, casting, …])	Element-wise minimum of array elements.
fmax(x1, x2, /[, out, where, casting, …])	Element-wise maximum of array elements.
fmin(x1, x2, /[, out, where, casting, …])	Element-wise minimum of array elements.

# Floating functions

isfinite(x, /[, out, where, casting, order, …])	Test element-wise for finiteness (not infinity or not Not a Number).
isinf(x, /[, out, where, casting, order, …])	Test element-wise for positive or negative infinity.
isnan(x, /[, out, where, casting, order, …])	Test element-wise for NaN and return result as a boolean array.
isnat(x, /[, out, where, casting, order, …])	Test element-wise for NaT (not a time) and return result as a boolean array.
fabs(x, /[, out, where, casting, order, …])	Compute the absolute values element-wise.
signbit(x, /[, out, where, casting, order, …])	Returns element-wise True where signbit is set (less than zero).
copysign(x1, x2, /[, out, where, casting, …])	Change the sign of x1 to that of x2, element-wise.
nextafter(x1, x2, /[, out, where, casting, …])	Return the next floating-point value after x1 towards x2, element-wise.
spacing(x, /[, out, where, casting, order, …])	Return the distance between x and the nearest adjacent number.
modf(x[, out1, out2], / [[, out, where, …])	Return the fractional and integral parts of an array, element-wise.
ldexp(x1, x2, /[, out, where, casting, …])	Returns x1 * 2**x2, element-wise.
frexp(x[, out1, out2], / [[, out, where, …])	Decompose the elements of x into mantissa and twos exponent.
fmod(x1, x2, /[, out, where, casting, …])	Return the element-wise remainder of division.
floor(x, /[, out, where, casting, order, …])	Return the floor of the input, element-wise.
ceil(x, /[, out, where, casting, order, …])	Return the ceiling of the input, element-wise.
trunc(x, /[, out, where, casting, order, …])	Return the truncated value of the input, element-wise.
 
"""


mafunc_str = """

# From existing data

ma.masked_array	alias of MaskedArray
ma.array(data[, dtype, copy, order, mask, …])	An array class with possibly masked values.
ma.copy(self, *args, **params) a.copy(order=)	Return a copy of the array.
ma.frombuffer(buffer[, dtype, count, offset])	Interpret a buffer as a 1-dimensional array.
ma.fromfunction(function, shape, **kwargs)	Construct an array by executing a function over each coordinate.
ma.MaskedArray.copy([order])	Return a copy of the array.

# Ones and zeros

ma.empty(shape[, dtype, order])	Return a new array of given shape and type, without initializing entries.
ma.empty_like(prototype[, dtype, order, subok])	Return a new array with the same shape and type as a given array.
ma.masked_all(shape[, dtype])	Empty masked array with all elements masked.
ma.masked_all_like(arr)	Empty masked array with the properties of an existing array.
ma.ones(shape[, dtype, order])	Return a new array of given shape and type, filled with ones.
ma.zeros(shape[, dtype, order])	Return a new array of given shape and type, filled with zeros.

# Inspecting the array

ma.all(self[, axis, out, keepdims])	Returns True if all elements evaluate to True.
ma.any(self[, axis, out, keepdims])	Returns True if any of the elements of a evaluate to True.
ma.count(self[, axis, keepdims])	Count the non-masked elements of the array along the given axis.
ma.count_masked(arr[, axis])	Count the number of masked elements along the given axis.
ma.getmask(a)	Return the mask of a masked array, or nomask.
ma.getmaskarray(arr)	Return the mask of a masked array, or full boolean array of False.
ma.getdata(a[, subok])	Return the data of a masked array as an ndarray.
ma.nonzero(self)	Return the indices of unmasked elements that are not zero.
ma.shape(obj)	Return the shape of an array.
ma.size(obj[, axis])	Return the number of elements along a given axis.
ma.is_masked(x)	Determine whether input has masked values.
ma.is_mask(m)	Return True if m is a valid, standard mask.
ma.MaskedArray.data	Return the current data, as a view of the original underlying data.
ma.MaskedArray.mask	Mask
ma.MaskedArray.recordmask	Return the mask of the records.
ma.MaskedArray.all([axis, out, keepdims])	Returns True if all elements evaluate to True.
ma.MaskedArray.any([axis, out, keepdims])	Returns True if any of the elements of a evaluate to True.
ma.MaskedArray.count([axis, keepdims])	Count the non-masked elements of the array along the given axis.
ma.MaskedArray.nonzero()	Return the indices of unmasked elements that are not zero.
ma.shape(obj)	Return the shape of an array.
ma.size(obj[, axis])	Return the number of elements along a given axis.

# Manipulating a MaskedArray

# Changing the shape

ma.ravel(self[, order])	Returns a 1D version of self, as a view.
ma.reshape(a, new_shape[, order])	Returns an array containing the same data with a new shape.
ma.resize(x, new_shape)	Return a new masked array with the specified size and shape.
ma.MaskedArray.flatten([order])	Return a copy of the array collapsed into one dimension.
ma.MaskedArray.ravel([order])	Returns a 1D version of self, as a view.
ma.MaskedArray.reshape(*s, **kwargs)	Give a new shape to the array without changing its data.
ma.MaskedArray.resize(newshape[, refcheck, …])	

# Modifying axes

ma.swapaxes(self, *args, …)	Return a view of the array with axis1 and axis2 interchanged.
ma.transpose(a[, axes])	Permute the dimensions of an array.
ma.MaskedArray.swapaxes(axis1, axis2)	Return a view of the array with axis1 and axis2 interchanged.
ma.MaskedArray.transpose(*axes)	Returns a view of the array with axes transposed.

# Changing the number of dimensions

ma.atleast_1d(*arys)	Convert inputs to arrays with at least one dimension.
ma.atleast_2d(*arys)	View inputs as arrays with at least two dimensions.
ma.atleast_3d(*arys)	View inputs as arrays with at least three dimensions.
ma.expand_dims(x, axis)	Expand the shape of an array.
ma.squeeze(a[, axis])	Remove single-dimensional entries from the shape of an array.
ma.MaskedArray.squeeze([axis])	Remove single-dimensional entries from the shape of a.
ma.stack(arrays[, axis, out])	Join a sequence of arrays along a new axis.
ma.column_stack(tup)	Stack 1-D arrays as columns into a 2-D array.
ma.concatenate(arrays[, axis])	Concatenate a sequence of arrays along the given axis.
ma.dstack(tup)	Stack arrays in sequence depth wise (along third axis).
ma.hstack(tup)	Stack arrays in sequence horizontally (column wise).
ma.hsplit(ary, indices_or_sections)	Split an array into multiple sub-arrays horizontally (column-wise).
ma.mr_	Translate slice objects to concatenation along the first axis.
ma.row_stack(tup)	Stack arrays in sequence vertically (row wise).
ma.vstack(tup)	Stack arrays in sequence vertically (row wise).

# Joining arrays

ma.stack(arrays[, axis, out])	Join a sequence of arrays along a new axis.
ma.column_stack(tup)	Stack 1-D arrays as columns into a 2-D array.
ma.concatenate(arrays[, axis])	Concatenate a sequence of arrays along the given axis.
ma.append(a, b[, axis])	Append values to the end of an array.
ma.dstack(tup)	Stack arrays in sequence depth wise (along third axis).
ma.hstack(tup)	Stack arrays in sequence horizontally (column wise).
ma.vstack(tup)	Stack arrays in sequence vertically (row wise).

# Operations on masks

# Creating a mask

ma.make_mask(m[, copy, shrink, dtype])	Create a boolean mask from an array.
ma.make_mask_none(newshape[, dtype])	Return a boolean mask of the given shape, filled with False.
ma.mask_or(m1, m2[, copy, shrink])	Combine two masks with the logical_or operator.
ma.make_mask_descr(ndtype)	Construct a dtype description list from a given dtype.

# Accessing a mask

ma.getmask(a)	Return the mask of a masked array, or nomask.
ma.getmaskarray(arr)	Return the mask of a masked array, or full boolean array of False.
ma.masked_array.mask	Mask

# Finding masked data

ma.flatnotmasked_contiguous(a)	Find contiguous unmasked data in a masked array along the given axis.
ma.flatnotmasked_edges(a)	Find the indices of the first and last unmasked values.
ma.notmasked_contiguous(a[, axis])	Find contiguous unmasked data in a masked array along the given axis.
ma.notmasked_edges(a[, axis])	Find the indices of the first and last unmasked values along an axis.
ma.clump_masked(a)	Returns a list of slices corresponding to the masked clumps of a 1-D array.
ma.clump_unmasked(a)	Return list of slices corresponding to the unmasked clumps of a 1-D array.

# Modifying a mask

ma.mask_cols(a[, axis])	Mask columns of a 2D array that contain masked values.
ma.mask_or(m1, m2[, copy, shrink])	Combine two masks with the logical_or operator.
ma.mask_rowcols(a[, axis])	Mask rows and/or columns of a 2D array that contain masked values.
ma.mask_rows(a[, axis])	Mask rows of a 2D array that contain masked values.
ma.harden_mask(self)	Force the mask to hard.
ma.soften_mask(self)	Force the mask to soft.
ma.MaskedArray.harden_mask()	Force the mask to hard.
ma.MaskedArray.soften_mask()	Force the mask to soft.
ma.MaskedArray.shrink_mask()	Reduce a mask to nomask when possible.
ma.MaskedArray.unshare_mask()	Copy the mask and set the sharedmask flag to False.

#Conversion operations

# > to a masked array

ma.asarray(a[, dtype, order])	Convert the input to a masked array of the given data-type.
ma.asanyarray(a[, dtype])	Convert the input to a masked array, conserving subclasses.
ma.fix_invalid(a[, mask, copy, fill_value])	Return input with invalid data masked and replaced by a fill value.
ma.masked_equal(x, value[, copy])	Mask an array where equal to a given value.
ma.masked_greater(x, value[, copy])	Mask an array where greater than a given value.
ma.masked_greater_equal(x, value[, copy])	Mask an array where greater than or equal to a given value.
ma.masked_inside(x, v1, v2[, copy])	Mask an array inside a given interval.
ma.masked_invalid(a[, copy])	Mask an array where invalid values occur (NaNs or infs).
ma.masked_less(x, value[, copy])	Mask an array where less than a given value.
ma.masked_less_equal(x, value[, copy])	Mask an array where less than or equal to a given value.
ma.masked_not_equal(x, value[, copy])	Mask an array where not equal to a given value.
ma.masked_object(x, value[, copy, shrink])	Mask the array x where the data are exactly equal to value.
ma.masked_outside(x, v1, v2[, copy])	Mask an array outside a given interval.
ma.masked_values(x, value[, rtol, atol, …])	Mask using floating point equality.
ma.masked_where(condition, a[, copy])	Mask an array where a condition is met.

# > to a ndarray

ma.compress_cols(a)	Suppress whole columns of a 2-D array that contain masked values.
ma.compress_rowcols(x[, axis])	Suppress the rows and/or columns of a 2-D array that contain masked values.
ma.compress_rows(a)	Suppress whole rows of a 2-D array that contain masked values.
ma.compressed(x)	Return all the non-masked data as a 1-D array.
ma.filled(a[, fill_value])	Return input as an array with masked data replaced by a fill value.
ma.MaskedArray.compressed()	Return all the non-masked data as a 1-D array.
ma.MaskedArray.filled([fill_value])	Return a copy of self, with masked values filled with a given value.

#> to another object

ma.MaskedArray.tofile(fid[, sep, format])	Save a masked array to a file in binary format.
ma.MaskedArray.tolist([fill_value])	Return the data portion of the masked array as a hierarchical Python list.
ma.MaskedArray.torecords()	Transforms a masked array into a flexible-type array.
ma.MaskedArray.tobytes([fill_value, order])	Return the array data as a string containing the raw bytes in the array.

# Pickling and unpickling

ma.dump(a, F)	Pickle a masked array to a file.
ma.dumps(a)	Return a string corresponding to the pickling of a masked array.
ma.load(F)	Wrapper around cPickle.load which accepts either a file-like object or a filename.
ma.loads(strg)	Load a pickle from the current string.

# Filling a masked array

ma.common_fill_value(a, b)	Return the common filling value of two masked arrays, if any.
ma.default_fill_value(obj)	Return the default fill value for the argument object.
ma.maximum_fill_value(obj)	Return the minimum value that can be represented by the dtype of an object.
ma.maximum_fill_value(obj)	Return the minimum value that can be represented by the dtype of an object.
ma.set_fill_value(a, fill_value)	Set the filling value of a, if a is a masked array.
ma.MaskedArray.get_fill_value()	Return the filling value of the masked array.
ma.MaskedArray.set_fill_value([value])	Set the filling value of the masked array.
ma.MaskedArray.fill_value	Filling value.
Masked arrays arithmetics

# Arithmetics

ma.anom(self[, axis, dtype])	Compute the anomalies (deviations from the arithmetic mean) along the given axis.
ma.anomalies(self[, axis, dtype])	Compute the anomalies (deviations from the arithmetic mean) along the given axis.
ma.average(a[, axis, weights, returned])	Return the weighted average of array over the given axis.
ma.conjugate(x, /[, out, where, casting, …])	Return the complex conjugate, element-wise.
ma.corrcoef(x[, y, rowvar, bias, …])	Return Pearson product-moment correlation coefficients.
ma.cov(x[, y, rowvar, bias, allow_masked, ddof])	Estimate the covariance matrix.
ma.cumsum(self[, axis, dtype, out])	Return the cumulative sum of the array elements over the given axis.
ma.cumprod(self[, axis, dtype, out])	Return the cumulative product of the array elements over the given axis.
ma.mean(self[, axis, dtype, out, keepdims])	Returns the average of the array elements along given axis.
ma.median(a[, axis, out, overwrite_input, …])	Compute the median along the specified axis.
ma.power(a, b[, third])	Returns element-wise base array raised to power from second array.
ma.prod(self[, axis, dtype, out, keepdims])	Return the product of the array elements over the given axis.
ma.std(self[, axis, dtype, out, ddof, keepdims])	Returns the standard deviation of the array elements along given axis.
ma.sum(self[, axis, dtype, out, keepdims])	Return the sum of the array elements over the given axis.
ma.var(self[, axis, dtype, out, ddof, keepdims])	Compute the variance along the specified axis.
ma.MaskedArray.anom([axis, dtype])	Compute the anomalies (deviations from the arithmetic mean) along the given axis.
ma.MaskedArray.cumprod([axis, dtype, out])	Return the cumulative product of the array elements over the given axis.
ma.MaskedArray.cumsum([axis, dtype, out])	Return the cumulative sum of the array elements over the given axis.
ma.MaskedArray.mean([axis, dtype, out, keepdims])	Returns the average of the array elements along given axis.
ma.MaskedArray.prod([axis, dtype, out, keepdims])	Return the product of the array elements over the given axis.
ma.MaskedArray.std([axis, dtype, out, ddof, …])	Returns the standard deviation of the array elements along given axis.
ma.MaskedArray.sum([axis, dtype, out, keepdims])	Return the sum of the array elements over the given axis.
ma.MaskedArray.var([axis, dtype, out, ddof, …])	Compute the variance along the specified axis.

# Minimum/maximum

ma.argmax(self[, axis, fill_value, out])	Returns array of indices of the maximum values along the given axis.
ma.argmin(self[, axis, fill_value, out])	Return array of indices to the minimum values along the given axis.
ma.max(obj[, axis, out, fill_value, keepdims])	Return the maximum along a given axis.
ma.min(obj[, axis, out, fill_value, keepdims])	Return the minimum along a given axis.
ma.ptp(obj[, axis, out, fill_value, keepdims])	Return (maximum - minimum) along the given dimension (i.e.
ma.MaskedArray.argmax([axis, fill_value, out])	Returns array of indices of the maximum values along the given axis.
ma.MaskedArray.argmin([axis, fill_value, out])	Return array of indices to the minimum values along the given axis.
ma.MaskedArray.max([axis, out, fill_value, …])	Return the maximum along a given axis.
ma.MaskedArray.min([axis, out, fill_value, …])	Return the minimum along a given axis.
ma.MaskedArray.ptp([axis, out, fill_value, …])	Return (maximum - minimum) along the given dimension (i.e.

# Sorting

ma.argsort(a[, axis, kind, order, endwith, …])	Return an ndarray of indices that sort the array along the specified axis.
ma.sort(a[, axis, kind, order, endwith, …])	Sort the array, in-place
ma.MaskedArray.argsort([axis, kind, order, …])	Return an ndarray of indices that sort the array along the specified axis.
ma.MaskedArray.sort([axis, kind, order, …])	Sort the array, in-place

# Algebra

ma.diag(v[, k])	Extract a diagonal or construct a diagonal array.
ma.dot(a, b[, strict, out])	Return the dot product of two arrays.
ma.identity(n[, dtype])	Return the identity array.
ma.inner(a, b)	Inner product of two arrays.
ma.innerproduct(a, b)	Inner product of two arrays.
ma.outer(a, b)	Compute the outer product of two vectors.
ma.outerproduct(a, b)	Compute the outer product of two vectors.
ma.trace(self[, offset, axis1, axis2, …])	Return the sum along diagonals of the array.
ma.transpose(a[, axes])	Permute the dimensions of an array.
ma.MaskedArray.trace([offset, axis1, axis2, …])	Return the sum along diagonals of the array.
ma.MaskedArray.transpose(*axes)	Returns a view of the array with axes transposed.

# Polynomial fit

ma.vander(x[, n])	Generate a Vandermonde matrix.
ma.polyfit(x, y, deg[, rcond, full, w, cov])	Least squares polynomial fit.

# Clipping and rounding

ma.around	Round an array to the given number of decimals.
ma.clip(a, a_min, a_max[, out])	Clip (limit) the values in an array.
ma.round(a[, decimals, out])	Return a copy of a, rounded to ‘decimals’ places.

# Miscellanea

ma.allequal(a, b[, fill_value])	Return True if all entries of a and b are equal, using fill_value as a truth value where either or both are masked.
ma.allclose(a, b[, masked_equal, rtol, atol])	Returns True if two arrays are element-wise equal within a tolerance.
ma.apply_along_axis(func1d, axis, arr, …)	Apply a function to 1-D slices along the given axis.
ma.arange([start,] stop[, step,][, dtype])	Return evenly spaced values within a given interval.
ma.choose(indices, choices[, out, mode])	Use an index array to construct a new array from a set of choices.
ma.ediff1d(arr[, to_end, to_begin])	Compute the differences between consecutive elements of an array.
ma.indices(dimensions[, dtype])	Return an array representing the indices of a grid.
ma.where(condition[, x, y])	Return a masked array with elements from x or y, depending on condition.

"""

def liste_ufunc():

    liste = ufunc_str.split("\n")

    ufuncs = []
    for item in liste:
        item = item.strip()
        if not item:
            continue
        if item.startswith('#'):
            continue
        item = item.split('(')
        ufuncs.append(item[0])
    return ufuncs

all_func_str = """

abs  ::  absolute(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
absolute  ::  absolute(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
add  ::  add(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
alen  ::   - Return the length of the first dimension of the input array.
all  ::   - Test whether all array elements along a given axis evaluate to True.
allclose  ::   - Returns True if two arrays are element-wise equal within a tolerance.
alltrue  ::   - Check if all elements of input array are true.
amax  ::   - Return the maximum of an array or maximum along an axis.
amin  ::   - Return the minimum of an array or minimum along an axis.
angle  ::   - Return the angle of the complex argument.
any  ::   - Test whether any array element along a given axis evaluates to True.
append  ::   - Append values to the end of an array.
apply_along_axis  ::   - Apply a function to 1-D slices along the given axis.
apply_over_axes  ::   - Apply a function repeatedly over multiple axes.
arange  ::  arange([start,] stop[, step,], dtype=None) - 
arccos  ::  arccos(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
arccosh  ::  arccosh(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
arcsin  ::  arcsin(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
arcsinh  ::  arcsinh(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
arctan  ::  arctan(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
arctan2  ::  arctan2(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
arctanh  ::  arctanh(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
argmax  ::   - Returns the indices of the maximum values along an axis.
argmin  ::   - Returns the indices of the minimum values along an axis.
argpartition  ::   - Perform an indirect partition along the given axis using the
argsort  ::   - Returns the indices that would sort an array.
argwhere  ::   - Find the indices of array elements that are non-zero, grouped by element.
around  ::   - Evenly round to the given number of decimals.
array_equal  ::   - True if two arrays have the same shape and elements, False otherwise.
array_equiv  ::   - Returns True if input arrays are shape consistent and all elements equal.
array_split  ::   - Split an array into multiple sub-arrays.
asanyarray  ::  Convert the input to an ndarray, but pass ndarray subclasses through. - 
asarray  ::  Convert the input to an array. - 
asarray_chkfinite  ::  Convert the input to an array, checking for NaNs or Infs. - 
ascontiguousarray  ::   - Return a contiguous array in memory (C order).
asfarray  ::   - Return an array converted to a float type.
asfortranarray  ::   - Return an array laid out in Fortran order in memory.
asmatrix  ::   - Interpret the input as a matrix.
asscalar  ::   - Convert an array of size 1 to its scalar equivalent.
atleast_1d  ::   - Convert inputs to arrays with at least one dimension.
atleast_2d  ::   - View inputs as arrays with at least two dimensions.
atleast_3d  ::   - View inputs as arrays with at least three dimensions.
average  ::   - Compute the weighted average along the specified axis.
bartlett  ::   - Return the Bartlett window.
blackman  ::   - Return the Blackman window.
block  ::   - Assemble an nd-array from nested lists of blocks.
bmat  ::   - Build a matrix object from a string, nested sequence, or array.
bool  ::  bool(x) -> bool - 
bool8  ::  NumPy's Boolean type.  Character code: ``?``.  Alias: bool8
broadcast  ::  Produce an object that mimics broadcasting. - 
broadcast_arrays  ::   - Broadcast any number of arrays against each other.
broadcast_to  ::  Broadcast an array to a new shape. - 
busday_count  ::  busday_count(begindates, enddates, weekmask='1111100', holidays=[], busdaycal=None, out=None) - 
busday_offset  ::  busday_offset(dates, offsets, roll='raise', weekmask='1111100', holidays=None, busdaycal=None, out=None) - 
busdaycalendar  ::  busdaycalendar(weekmask='1111100', holidays=None) - 
byte  ::  8-bit integer. Character code ``b``. C char compatible.
byte_bounds  ::   - Returns pointers to the end-points of an array.
can_cast  ::  can_cast(from_, to, casting='safe') - 
cast  ::   - Base object for a dictionary for look-up with any alias for an array dtype.
cbrt  ::  cbrt(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
cdouble  ::  Composed of two 64 bit floats
ceil  ::  ceil(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
cfloat  ::  Composed of two 64 bit floats
char  ::   - This module contains a set of functions for vectorized string
chararray  ::   - chararray(shape, itemsize=1, unicode=False, buffer=None, offset=0,
choose  ::   - Construct an array from an index array and a set of arrays to choose from.
clip  ::   - Clip (limit) the values in an array.
clongdouble  ::  Composed of two 128 bit floats
clongfloat  ::  Composed of two 128 bit floats
column_stack  ::   - Stack 1-D arrays as columns into a 2-D array.
common_type  ::   - Return a scalar type which is common to the input arrays.
compat  ::   - Compatibility module.
complex  ::  complex(real[, imag]) -> complex number - 
complex128  ::  Composed of two 64 bit floats
complex256  ::  Composed of two 128 bit floats
complex64  ::  Composed of two 32 bit floats
compress  ::   - Return selected slices of an array along given axis.
concatenate  ::  concatenate((a1, a2, ...), axis=0, out=None) - 
conj  ::  conjugate(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
conjugate  ::  conjugate(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
convolve  ::   - Returns the discrete, linear convolution of two one-dimensional sequences.
copy  ::   - Return an array copy of the given object.
copysign  ::  copysign(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
copyto  ::  copyto(dst, src, casting='same_kind', where=True) - 
core  ::  Defines a multi-dimensional array and useful procedures for Numerical computation. - 
corrcoef  ::   - Return Pearson product-moment correlation coefficients.
correlate  ::   - Cross-correlation of two 1-dimensional sequences.
cos  ::  cos(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
cosh  ::  cosh(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
count_nonzero  ::   - Counts the number of non-zero values in the array ``a``.
cov  ::   - Estimate a covariance matrix, given data and weights.
cross  ::   - Return the cross product of two (arrays of) vectors.
csingle  ::  Composed of two 32 bit floats
ctypeslib  ::   - ============================
cumprod  ::   - Return the cumulative product of elements along a given axis.
cumproduct  ::   - Return the cumulative product over the given axis.
cumsum  ::   - Return the cumulative sum of the elements along a given axis.
datetime_as_string  ::  datetime_as_string(arr, unit=None, timezone='naive', casting='same_kind') - 
datetime_data  ::  datetime_data(dtype, /) - 
deg2rad  ::  deg2rad(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
degrees  ::  degrees(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
delete  ::   - Return a new array with sub-arrays along an axis deleted. For a one
deprecate  ::   - Issues a DeprecationWarning, adds warning to `old_name`'s
diag  ::   - Extract a diagonal or construct a diagonal array.
diag_indices  ::   - Return the indices to access the main diagonal of an array.
diag_indices_from  ::   - Return the indices to access the main diagonal of an n-dimensional array.
diagflat  ::   - Create a two-dimensional array with the flattened input as a diagonal.
diagonal  ::   - Return specified diagonals.
diff  ::   - Calculate the n-th discrete difference along the given axis.
digitize  ::  digitize(x, bins, right=False) - 
disp  ::   - Display a message on a device.
divide  ::  true_divide(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
divmod  ::  divmod(x1, x2[, out1, out2], / [, out=(None, None)], *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
dot  ::  dot(a, b, out=None) - 
double  ::  64-bit floating-point number. Character code 'd'. Python float compatible.
dsplit  ::   - Split array into multiple sub-arrays along the 3rd axis (depth).
dstack  ::   - Stack arrays in sequence depth wise (along third axis).
dtype  ::  dtype(obj, align=False, copy=False) - 
e  ::  float(x) -> floating point number - 
ediff1d  ::   - The differences between consecutive elements of an array.
einsum  ::   - einsum(subscripts, *operands, out=None, dtype=None, order='K',
einsum_path  ::   - einsum_path(subscripts, *operands, optimize='greedy')
emath  ::   - Wrapper functions to more user-friendly calling of certain math functions
empty  ::  empty(shape, dtype=float, order='C') - 
empty_like  ::  empty_like(prototype, dtype=None, order='K', subok=True) - 
equal  ::  equal(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
erf  ::  erf(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
errstate  ::   - errstate(**kwargs)
euler_gamma  ::  float(x) -> floating point number - 
exp  ::  exp(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
exp2  ::  exp2(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
expand_dims  ::   - Expand the shape of an array.
expm1  ::  expm1(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
extract  ::   - Return the elements of an array that satisfy some condition.
eye  ::   - Return a 2-D array with ones on the diagonal and zeros elsewhere.
fabs  ::  fabs(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
fastCopyAndTranspose  ::  _fastCopyAndTranspose(a)
fft  ::   - Discrete Fourier Transform (:mod:`numpy.fft`)
fill_diagonal  ::  Fill the main diagonal of the given array of any dimensionality. - 
find_common_type  ::   - Determine common type following standard coercion rules.
finfo  ::   - finfo(dtype)
fix  ::   - Round to nearest integer towards zero.
flatiter  ::  Flat iterator object to iterate over arrays. - 
flatnonzero  ::   - Return indices that are non-zero in the flattened version of a.
flip  ::   - Reverse the order of elements in an array along the given axis.
fliplr  ::   - Flip array in the left/right direction.
flipud  ::   - Flip array in the up/down direction.
float  ::  float(x) -> floating point number - 
float128  ::  128-bit floating-point number. Character code: 'g'. C long float - compatible.
float32  ::  32-bit floating-point number. Character code 'f'. C float compatible.
float64  ::  64-bit floating-point number. Character code 'd'. Python float compatible.
float_power  ::  float_power(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
floor  ::  floor(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
floor_divide  ::  floor_divide(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
fmax  ::  fmax(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
fmin  ::  fmin(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
fmod  ::  fmod(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
format_float_positional  ::   - Format a floating-point scalar as a decimal string in positional notation.
format_float_scientific  ::   - Format a floating-point scalar as a decimal string in scientific notation.
format_parser  ::   - Class to convert formats, names, titles description to a dtype.
frexp  ::  frexp(x[, out1, out2], / [, out=(None, None)], *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
frombuffer  ::  frombuffer(buffer, dtype=float, count=-1, offset=0) - 
fromfile  ::  fromfile(file, dtype=float, count=-1, sep='') - 
fromfunction  ::   - Construct an array by executing a function over each coordinate.
fromiter  ::  fromiter(iterable, dtype, count=-1) - 
frompyfunc  ::  frompyfunc(func, nin, nout) - 
fromregex  ::   - Construct an array from a text file, using regular expression parsing.
fromstring  ::  fromstring(string, dtype=float, count=-1, sep='') - 
full  ::   - Return a new array of given shape and type, filled with `fill_value`.
full_like  ::   - Return a full array with the same shape and type as a given array.
fv  ::   - Compute the future value.
gcd  ::  gcd(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
generic  ::  Base class for numpy scalar types. - 
genfromtxt  ::   - Load data from a text file, with missing values handled as specified.
geomspace  ::   - Return numbers spaced evenly on a log scale (a geometric progression).
get_array_wrap  ::  Find the wrapper for the array with the highest priority. - 
get_include  ::   - Return the directory that contains the NumPy \*.h header files.
get_printoptions  ::   - Return the current print options.
getbufsize  ::   - Return the size of the buffer used in ufuncs.
geterr  ::   - Get the current way of handling floating-point errors.
geterrcall  ::   - Return the current callback function used on floating-point errors.
geterrobj  ::  geterrobj() - 
gradient  ::   - Return the gradient of an N-dimensional array.
greater  ::  greater(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
greater_equal  ::  greater_equal(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
hamming  ::   - Return the Hamming window.
hanning  ::   - Return the Hanning window.
heaviside  ::  heaviside(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
histogram  ::   - Compute the histogram of a set of data.
histogram2d  ::   - Compute the bi-dimensional histogram of two data samples.
histogram_bin_edges  ::   - Function to calculate only the edges of the bins used by the `histogram` function.
histogramdd  ::   - Compute the multidimensional histogram of some data.
hsplit  ::   - Split an array into multiple sub-arrays horizontally (column-wise).
hstack  ::   - Stack arrays in sequence horizontally (column wise).
hypot  ::  hypot(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
i0  ::   - Modified Bessel function of the first kind, order 0.
identity  ::   - Return the identity array.
iinfo  ::   - iinfo(type)
imag  ::   - Return the imaginary part of the complex argument.
in1d  ::   - Test whether each element of a 1-D array is also present in a second array.
index_exp  ::   - A nicer way to build up index tuples for arrays.
indices  ::   - Return an array representing the indices of a grid.
inf  ::  float(x) -> floating point number - 
info  ::   - Get help information for a function, class, or module.
infty  ::  float(x) -> floating point number - 
inner  ::  inner(a, b) - 
insert  ::   - Insert values along the given axis before the given indices.
int  ::  int(x=0) -> integer - int(x, base=10) -> integer
int0  ::  64-bit integer. Character code 'l'. Python int compatible.
int16  ::  16-bit integer. Character code ``h``. C short compatible.
int32  ::  32-bit integer. Character code 'i'. C int compatible.
int64  ::  64-bit integer. Character code 'l'. Python int compatible.
int8  ::  8-bit integer. Character code ``b``. C char compatible.
intc  ::  32-bit integer. Character code 'i'. C int compatible.
interp  ::   - One-dimensional linear interpolation.
intersect1d  ::   - Find the intersection of two arrays.
intp  ::  64-bit integer. Character code 'l'. Python int compatible.
invert  ::  invert(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
ipmt  ::   - Compute the interest portion of a payment.
irr  ::   - Return the Internal Rate of Return (IRR).
is_busday  ::  is_busday(dates, weekmask='1111100', holidays=None, busdaycal=None, out=None) - 
isclose  ::   - Returns a boolean array where two arrays are element-wise equal within a
iscomplex  ::   - Returns a bool array, where True if input element is complex.
iscomplexobj  ::   - Check for a complex type or an array of complex numbers.
isfinite  ::  isfinite(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
isfortran  ::   - Returns True if the array is Fortran contiguous but *not* C contiguous.
isin  ::   - Calculates `element in test_elements`, broadcasting over `element` only.
isinf  ::  isinf(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
isnan  ::  isnan(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
isnat  ::  isnat(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
isneginf  ::   - Test element-wise for negative infinity, return result as bool array.
isposinf  ::   - Test element-wise for positive infinity, return result as bool array.
isreal  ::   - Returns a bool array, where True if input element is real.
isrealobj  ::   - Return True if x is a not complex type or an array of complex numbers.
isscalar  ::   - Returns True if the type of `num` is a scalar type.
issctype  ::   - Determines whether the given object represents a scalar data-type.
issubdtype  ::   - Returns True if first argument is a typecode lower/equal in type hierarchy.
issubsctype  ::   - Determine if the first argument is a subclass of the second argument.
iterable  ::   - Check whether or not an object can be iterated over.
kaiser  ::   - Return the Kaiser window.
kron  ::   - Kronecker product of two arrays.
lcm  ::  lcm(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
ldexp  ::  ldexp(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
left_shift  ::  left_shift(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
less  ::  less(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
less_equal  ::  less_equal(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
lexsort  ::  lexsort(keys, axis=-1) - 
lib  ::   - Basic functions used by several sub-packages and
linalg  ::  Core Linear Algebra Tools - -------------------------
linspace  ::   - Return evenly spaced numbers over a specified interval.
little_endian  ::  bool(x) -> bool - 
load  ::   - Load arrays or pickled objects from ``.npy``, ``.npz`` or pickled files.
loadtxt  ::   - Load data from a text file.
log  ::  log(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
log10  ::  log10(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
log1p  ::  log1p(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
log2  ::  log2(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
logaddexp  ::  logaddexp(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
logaddexp2  ::  logaddexp2(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
logical_and  ::  logical_and(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
logical_not  ::  logical_not(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
logical_or  ::  logical_or(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
logical_xor  ::  logical_xor(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
logspace  ::   - Return numbers spaced evenly on a log scale.
long  ::  int(x=0) -> integer - int(x, base=10) -> integer
longcomplex  ::  Composed of two 128 bit floats
longdouble  ::  128-bit floating-point number. Character code: 'g'. C long float - compatible.
longfloat  ::  128-bit floating-point number. Character code: 'g'. C long float - compatible.
lookfor  ::   - Do a keyword search on docstrings.
ma  ::   - =============
mafromtxt  ::   - Load ASCII data stored in a text file and return a masked array.
mask_indices  ::   - Return the indices to access (n, n) arrays, given a masking function.
mat  ::   - Interpret the input as a matrix.
math  ::  This module is always available.  It provides access to the - mathematical functions defined by the C standard.
matmul  ::  matmul(a, b, out=None) - 
matrix  ::   - matrix(data, dtype=None, copy=True)
matrixlib  ::  Sub-package containing the matrix class and related functions. - 
max  ::   - Return the maximum of an array or maximum along an axis.
maximum  ::  maximum(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
maximum_sctype  ::   - Return the scalar type of highest precision of the same kind as the input.
may_share_memory  ::  may_share_memory(a, b, max_work=None) - 
mean  ::   - Compute the arithmetic mean along the specified axis.
median  ::   - Compute the median along the specified axis.
memmap  ::  Create a memory-map to an array stored in a *binary* file on disk. - 
meshgrid  ::   - Return coordinate matrices from coordinate vectors.
mgrid  ::  `nd_grid` instance which returns a dense multi-dimensional "meshgrid". - 
min  ::   - Return the minimum of an array or minimum along an axis.
min_scalar_type  ::  min_scalar_type(a) - 
minimum  ::  minimum(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
mintypecode  ::   - Return the character for the minimum-size type to which given types can
mirr  ::   - Modified internal rate of return.
mod  ::  remainder(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
modf  ::  modf(x[, out1, out2], / [, out=(None, None)], *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
moveaxis  ::   - Move axes of an array to new positions.
msort  ::   - Return a copy of an array sorted along the first axis.
multiply  ::  multiply(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
nan  ::  float(x) -> floating point number - 
nan_to_num  ::   - Replace NaN with zero and infinity with large finite numbers.
nanargmax  ::   - Return the indices of the maximum values in the specified axis ignoring
nanargmin  ::   - Return the indices of the minimum values in the specified axis ignoring
nancumprod  ::   - Return the cumulative product of array elements over a given axis treating Not a
nancumsum  ::   - Return the cumulative sum of array elements over a given axis treating Not a
nanmax  ::   - Return the maximum of an array or maximum along an axis, ignoring any
nanmean  ::   - Compute the arithmetic mean along the specified axis, ignoring NaNs.
nanmedian  ::   - Compute the median along the specified axis, while ignoring NaNs.
nanmin  ::   - Return minimum of an array or minimum along an axis, ignoring any NaNs.
nanpercentile  ::   - Compute the qth percentile of the data along the specified axis,
nanprod  ::   - Return the product of array elements over a given axis treating Not a
nanquantile  ::   - Compute the qth quantile of the data along the specified axis,
nanstd  ::   - Compute the standard deviation along the specified axis, while
nansum  ::   - Return the sum of array elements over a given axis treating Not a
nanvar  ::   - Compute the variance along the specified axis, while ignoring NaNs.
nbytes  ::   - Base object for a dictionary for look-up with any alias for an array dtype.
ndarray  ::  ndarray(shape, dtype=float, buffer=None, offset=0, - strides=None, order=None)
ndenumerate  ::   - Multidimensional index iterator.
ndfromtxt  ::   - Load ASCII data stored in a file and return it as a single array.
ndim  ::   - Return the number of dimensions of an array.
ndindex  ::   - An N-dimensional iterator object to index arrays.
nditer  ::  Efficient multi-dimensional iterator object to iterate over arrays. - To get started using this object, see the
negative  ::  negative(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
nested_iters  ::  Create nditers for use in nested loops - 
nextafter  ::  nextafter(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
nonzero  ::   - Return the indices of the elements that are non-zero.
not_equal  ::  not_equal(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
nper  ::   - Compute the number of periodic payments.
npv  ::   - Returns the NPV (Net Present Value) of a cash flow series.
numarray  ::  str(object='') -> str - str(bytes_or_buffer[, encoding[, errors]]) -> str
obj2sctype  ::   - Return the scalar dtype or NumPy equivalent of Python type of an object.
object  ::  The most base type
object0  ::  Any Python object.  Character code: 'O'.
ogrid  ::  `nd_grid` instance which returns an open multi-dimensional "meshgrid". - 
oldnumeric  ::  str(object='') -> str - str(bytes_or_buffer[, encoding[, errors]]) -> str
ones  ::   - Return a new array of given shape and type, filled with ones.
ones_like  ::   - Return an array of ones with the same shape and type as a given array.
outer  ::   - Compute the outer product of two vectors.
packbits  ::  packbits(myarray, axis=None) - 
pad  ::   - Pads an array.
partition  ::   - Return a partitioned copy of an array.
percentile  ::   - Compute the qth percentile of the data along the specified axis.
pi  ::  float(x) -> floating point number - 
piecewise  ::   - Evaluate a piecewise-defined function.
pkgload  ::  Load one or more packages into parent package top-level namespace. - 
place  ::   - Change elements of an array based on conditional and input values.
pmt  ::   - Compute the payment against loan principal plus interest.
poly  ::   - Find the coefficients of a polynomial with the given sequence of roots.
poly1d  ::   - A one-dimensional polynomial class.
polyadd  ::   - Find the sum of two polynomials.
polyder  ::   - Return the derivative of the specified order of a polynomial.
polydiv  ::   - Returns the quotient and remainder of polynomial division.
polyfit  ::   - Least squares polynomial fit.
polyint  ::   - Return an antiderivative (indefinite integral) of a polynomial.
polymul  ::   - Find the product of two polynomials.
polynomial  ::   - A sub-package for efficiently dealing with polynomials.
polysub  ::   - Difference (subtraction) of two polynomials.
polyval  ::   - Evaluate a polynomial at specific values.
positive  ::  positive(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
power  ::  power(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
ppmt  ::   - Compute the payment against loan principal.
printoptions  ::  Context manager for setting print options. - 
prod  ::   - Return the product of array elements over a given axis.
product  ::   - Return the product of array elements over a given axis.
promote_types  ::  promote_types(type1, type2) - 
ptp  ::   - Range of values (maximum - minimum) along an axis.
put  ::   - Replaces specified elements of an array with given values.
put_along_axis  ::   - Put values into the destination array by matching 1d index and data slices.
putmask  ::  putmask(a, mask, values) - 
pv  ::   - Compute the present value.
quantile  ::   - Compute the `q`th quantile of the data along the specified axis.
rad2deg  ::  rad2deg(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
radians  ::  radians(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
random  ::   - ========================
rank  ::   - Return the number of dimensions of an array.
rate  ::   - Compute the rate of interest per period.
ravel  ::  Return a contiguous flattened array. - 
ravel_multi_index  ::  ravel_multi_index(multi_index, dims, mode='raise', order='C') - 
real  ::   - Return the real part of the complex argument.
real_if_close  ::   - If complex input returns a real array if complex parts are close to zero.
rec  ::   - Record Arrays
recarray  ::  Construct an ndarray that allows field access using attributes. - 
recfromcsv  ::   - Load ASCII data stored in a comma-separated file.
recfromtxt  ::   - Load ASCII data from a file and return it in a record array.
reciprocal  ::  reciprocal(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
record  ::  A data-type scalar that allows field access as attribute lookup. - 
remainder  ::  remainder(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
repeat  ::   - Repeat elements of an array.
require  ::   - Return an ndarray of the provided type that satisfies requirements.
reshape  ::   - Gives a new shape to an array without changing its data.
resize  ::   - Return a new array with the specified shape.
result_type  ::  result_type(*arrays_and_dtypes) - 
right_shift  ::  right_shift(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
rint  ::  rint(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
roll  ::   - Roll array elements along a given axis.
rollaxis  ::   - Roll the specified axis backwards, until it lies in a given position.
roots  ::   - Return the roots of a polynomial with coefficients given in p.
rot90  ::   - Rotate an array by 90 degrees in the plane specified by axes.
round  ::   - Round an array to the given number of decimals.
row_stack  ::   - Stack arrays in sequence vertically (row wise).
safe_eval  ::   - Protected string evaluation.
save  ::   - Save an array to a binary file in NumPy ``.npy`` format.
savetxt  ::   - Save an array to a text file.
savez  ::   - Save several arrays into a single file in uncompressed ``.npz`` format.
savez_compressed  ::   - Save several arrays into a single file in compressed ``.npz`` format.
sctype2char  ::   - Return the string representation of a scalar dtype.
sctypeDict  ::  dict() -> new empty dictionary - dict(mapping) -> new dictionary initialized from a mapping object's
sctypeNA  ::  dict() -> new empty dictionary - dict(mapping) -> new dictionary initialized from a mapping object's
sctypes  ::  dict() -> new empty dictionary - dict(mapping) -> new dictionary initialized from a mapping object's
searchsorted  ::   - Find indices where elements should be inserted to maintain order.
select  ::   - Return an array drawn from elements in choicelist, depending on conditions.
set_numeric_ops  ::  set_numeric_ops(op1=func1, op2=func2, ...) - 
set_printoptions  ::   - Set printing options.
set_string_function  ::   - Set a Python function to be used when pretty printing arrays.
setbufsize  ::   - Set the size of the buffer used in ufuncs.
setdiff1d  ::   - Find the set difference of two arrays.
seterr  ::   - Set how floating-point errors are handled.
seterrcall  ::   - Set the floating-point error callback function or log object.
seterrobj  ::  seterrobj(errobj) - 
setxor1d  ::   - Find the set exclusive-or of two arrays.
shape  ::   - Return the shape of an array.
shares_memory  ::  shares_memory(a, b, max_work=None) - 
short  ::  16-bit integer. Character code ``h``. C short compatible.
sign  ::  sign(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
signbit  ::  signbit(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
sin  ::  sin(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
sinc  ::   - Return the sinc function.
single  ::  32-bit floating-point number. Character code 'f'. C float compatible.
singlecomplex  ::  Composed of two 32 bit floats
sinh  ::  sinh(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
size  ::   - Return the number of elements along a given axis.
sometrue  ::   - Check whether some values are true.
sort  ::   - Return a sorted copy of an array.
sort_complex  ::   - Sort a complex array using the real part first, then the imaginary part.
source  ::   - Print or write to a file the source code for a NumPy object.
spacing  ::  spacing(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
split  ::   - Split an array into multiple sub-arrays.
sqrt  ::  sqrt(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
square  ::  square(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
squeeze  ::   - Remove single-dimensional entries from the shape of an array.
stack  ::   - Join a sequence of arrays along a new axis.
std  ::   - Compute the standard deviation along the specified axis.
str  ::  str(object='') -> str - str(bytes_or_buffer[, encoding[, errors]]) -> str
subtract  ::  subtract(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
sum  ::   - Sum of array elements over a given axis.
swapaxes  ::   - Interchange two axes of an array.
sys  ::  This module provides access to some objects used or maintained by the - interpreter and to functions that interact strongly with the interpreter.
take  ::   - Take elements from an array along an axis.
take_along_axis  ::   - Take values from the input array by matching 1d index and data slices.
tan  ::  tan(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
tanh  ::  tanh(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
tensordot  ::   - Compute tensor dot product along specified axes for arrays >= 1-D.
test  ::   - Pytest test runner.
testing  ::  Common test support for all numpy test scripts. - 
tile  ::   - Construct an array by repeating A the number of times given by reps.
trace  ::   - Return the sum along diagonals of the array.
tracemalloc_domain  ::  int(x=0) -> integer - int(x, base=10) -> integer
transpose  ::   - Permute the dimensions of an array.
trapz  ::   - Integrate along the given axis using the composite trapezoidal rule.
tri  ::   - An array with ones at and below the given diagonal and zeros elsewhere.
tril  ::   - Lower triangle of an array.
tril_indices  ::   - Return the indices for the lower-triangle of an (n, m) array.
tril_indices_from  ::   - Return the indices for the lower-triangle of arr.
trim_zeros  ::   - Trim the leading and/or trailing zeros from a 1-D array or sequence.
triu  ::   - Upper triangle of an array.
triu_indices  ::   - Return the indices for the upper-triangle of an (n, m) array.
triu_indices_from  ::   - Return the indices for the upper-triangle of arr.
true_divide  ::  true_divide(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
trunc  ::  trunc(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) - 
typeDict  ::  dict() -> new empty dictionary - dict(mapping) -> new dictionary initialized from a mapping object's
typeNA  ::  dict() -> new empty dictionary - dict(mapping) -> new dictionary initialized from a mapping object's
typecodes  ::  dict() -> new empty dictionary - dict(mapping) -> new dictionary initialized from a mapping object's
typename  ::   - Return a description for the given data type code.
ufunc  ::  Functions that operate element by element on whole arrays. - 
unicode  ::  str(object='') -> str - str(bytes_or_buffer[, encoding[, errors]]) -> str
union1d  ::   - Find the union of two arrays.
unique  ::   - Find the unique elements of an array.
unpackbits  ::  unpackbits(myarray, axis=None) - 
unravel_index  ::  unravel_index(indices, dims, order='C') - 
unwrap  ::   - Unwrap by changing deltas between values to 2*pi complement.
vander  ::   - Generate a Vandermonde matrix.
var  ::   - Compute the variance along the specified axis.
vdot  ::  vdot(a, b) - 
vectorize  ::   - vectorize(pyfunc, otypes=None, doc=None, excluded=None, cache=False,
vsplit  ::   - Split an array into multiple sub-arrays vertically (row-wise).
vstack  ::   - Stack arrays in sequence vertically (row wise).
where  ::  where(condition, [x, y]) - 
who  ::   - Print the NumPy arrays in the given dictionary.
zeros  ::  zeros(shape, dtype=float, order='C') - 
zeros_like  ::   - Return an array of zeros with the same shape and type as a given array.

"""

if __name__ == '__main__':

    ufuncs = liste_ufunc()

    print (len(ufuncs), ufuncs)

    # import numpy as np
    # for item in dir(np):
    #     if item.startswith('_') or item.endswith('_'):
    #         continue
    #     f = getattr(np,item)
    #     _doc = f.__doc__.split('\n') if f.__doc__ else None
    #     if not _doc:
    #         continue
    #     l = min(len(_doc),2)
    #     doc = []
    #     for i in _doc[:l]:
    #         doc.append(i.strip())
    #     doc = " - ".join(doc)
    #     print(item, " :: ", doc)