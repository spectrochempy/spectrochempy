"""NDMathQuaternion class - Extension of NDMath with quaternion support."""

# Standard library imports
import copy as cpy

# Third-party imports
import numpy as np

# Local imports
from spectrochempy.core.dataset.arraymixins.ndmath import NDMath
from spectrochempy.utils.constants import NOMASK

# Try to load quaternion support
# -----------------------------------------------------------------------------------
try:
    import quaternion

    typequaternion = np.quaternion
except ImportError:
    quaternion = None
    typequaternion = None

# Define quaternion utility functions if library is available
# -----------------------------------------------------------------------------------
if quaternion is not None:

    def as_quaternion(real_part, imag_part):
        """
        Convert complex array to quaternion.

        Parameters
        ----------
        real_part : ndarray
            Real parts (will be placed in quaternion components w and i)
        imag_part : ndarray
            Imaginary parts (will be placed in quaternion components j and k)

        Returns
        -------
        q : ndarray
            Array of quaternions
        """
        shape = real_part.shape
        dtype = np.result_type(real_part, imag_part)

        components = np.zeros(shape + (4,), dtype=dtype)
        components[..., 0] = real_part.real
        components[..., 1] = real_part.imag
        components[..., 2] = imag_part.real
        components[..., 3] = imag_part.imag

        return quaternion.as_quat_array(components)

    def as_float_array(q):
        """
        Convert quaternion array to float array with shape (..., 4).

        Parameters
        ----------
        q : ndarray
            Array of quaternions

        Returns
        -------
        components : ndarray
            Array with quaternion components as float array
        """
        return quaternion.as_float_array(q)

    def quat_as_complex_array(q):
        """
        Convert quaternion array to a pair of complex arrays.

        Parameters
        ----------
        q : ndarray
            Array of quaternions

        Returns
        -------
        real_part : ndarray
            Complex array with w and i components
        imag_part : ndarray
            Complex array with j and k components
        """
        components = quaternion.as_float_array(q)
        real_part = components[..., 0] + 1j * components[..., 1]
        imag_part = components[..., 2] + 1j * components[..., 3]
        return real_part, imag_part


class NDMathQuaternion(NDMath):
    """
    Extension of NDMath with quaternion support.

    This class adds quaternion-specific functionality to the NDMath class.
    It should be used when working with quaternion data in SpectroChemPy.
    """

    # Class constants
    # Set the quaternion-aware operations
    __quaternion_aware = [
        "add",
        "iadd",
        "sub",
        "isub",
        "mul",
        "imul",
        "div",
        "idiv",
        "log",
        "exp",
        "power",
        "negative",
        "conjugate",
        "copysign",
        "equal",
        "not_equal",
        "less",
        "less_equal",
        "isnan",
        "isinf",
        "isfinite",
        "absolute",
        "abs",
    ]

    @property
    def is_quaternion(self):
        """Check if data is quaternion type."""
        return self.dtype is typequaternion

    @_from_numpy_method
    def absolute(cls, dataset, dtype=None):
        r"""
        Calculate the absolute value of the given NDDataset element-wise.

        For quaternion input q=w+xi+yj+zk, the absolute value is
        :math:`\sqrt{w^2 + x^2 + y^2 + z^2}`.

        Parameters
        ----------
        dataset : `NDDataset` or :term:`array-like`
            Input array or object that can be converted to an array.
        dtype : dtype
            The type of the output array. If dtype is not given, infer the data type
            from the other input arguments.

        Returns
        -------
        `~spectrochempy.core.dataset.nddataset.NDDataset`
            The absolute value of each element in dataset.
        """
        if not cls.has_complex_dims:
            data = np.ma.fabs(
                dataset,
                dtype=dtype,
            )  # not a complex, return fabs should be faster

        elif not cls.is_quaternion:
            data = np.ma.sqrt(dataset.real**2 + dataset.imag**2)

        else:
            data = np.ma.sqrt(
                dataset.real**2
                + dataset.part("IR") ** 2
                + dataset.part("RI") ** 2
                + dataset.part("II") ** 2,
                dtype=dtype,
            )
            cls._is_quaternion = False

        cls._data = data.data
        cls._mask = data.mask

        return cls

    @_from_numpy_method
    def conjugate(cls, dataset, dim="x"):
        """
        Conjugate of the NDDataset in the specified dimension.

        For quaternion data, the conjugate is defined as q* = w-xi-yj-zk.

        Parameters
        ----------
        dataset : array_like
            Input array or object that can be converted to an array.
        dim : int, str, optional, default=(0,)
            Dimension names or indexes along which the method should be applied.

        Returns
        -------
        conjugated
            Same object or a copy depending on the `inplace` flag.

        See Also
        --------
        conj, real, imag, RR, RI, IR, II, part, set_complex, is_complex
        """
        axis, dim = cls.get_axis(dim, allows_none=True)

        if cls.is_quaternion:
            # For quaternion data, negate the imaginary parts
            dataset = dataset.swapdims(axis, -1)
            dataset[..., 1::2] = -dataset[..., 1::2]
            dataset = dataset(axis, -1)
        else:
            dataset = np.ma.conjugate(dataset)

        cls._data = dataset.data
        cls._mask = dataset.mask

        return cls

    def _preprocess_op_inputs(self, fname, inputs):
        inputs = list(inputs)  # work with a list of objs not tuples

        # By default the type of the result is set regarding the first obj in inputs
        # (except for some ufuncs that can return numpy arrays or masked numpy arrays
        # but sometimes we have something such as 2 * nd where nd is a NDDataset: In
        # this case we expect a dataset.

        # For binary function, we also determine if the function needs object with
        # compatible units.
        # If the object are not compatible then we raise an error

        # Take the objects out of the input list and get their types and units.
        # Additionally determine if we need to
        # use operation on masked arrays and/or on quaternion

        is_masked = False
        objtypes = []
        objunits = OrderedSet()
        returntype = None

        is_quaternion = False
        compatible_units = fname in self.__compatible_units

        for _i, obj in enumerate(inputs):
            # type
            objtype = type(obj).__name__
            objtypes.append(objtype)

            # units
            if hasattr(obj, "units"):
                objunits.add(ur.get_dimensionality(obj.units))
                if len(objunits) > 1 and compatible_units:
                    objunits = list(objunits)
                    raise DimensionalityError(
                        *objunits[::-1],
                        extra_msg=f", Units must be compatible "
                        f"for the `{fname}` operator",
                    )

            # returntype
            if objtype == "NDDataset":
                returntype = "NDDataset"
            elif objtype == "Coord" and returntype != "NDDataset":
                returntype = "Coord"
            else:
                # only the three above type have math capabilities in spectrochempy.
                pass

            # Do we have to deal with mask?
            if hasattr(obj, "mask") and np.any(obj.mask):
                is_masked = True

            # If one of the input is hypercomplex, this will demand a special treatment
            is_quaternion = (
                is_quaternion or False
                if not hasattr(obj, "is_quaternion")
                else obj.is_quaternion
            )

        # it may be necessary to change the object order regarding the types
        if returntype in ["NDDataset", "Coord"] and objtypes[0] != returntype:
            inputs.reverse()
            objtypes.reverse()

            if fname in ["mul", "multiply", "add", "iadd"]:
                pass
            elif fname in ["truediv", "divide", "true_divide"]:
                fname = "multiply"
                inputs[0] = np.reciprocal(inputs[0])
            elif fname in ["isub", "sub", "subtract"]:
                fname = "add"
                inputs[0] = np.negative(inputs[0])
            else:
                raise NotImplementedError

        return fname, inputs, objtypes, returntype, is_masked, is_quaternion

    def _op(
        self,
        f: Callable,
        inputs: Sequence[ArrayLike],
        isufunc: bool = False,
    ) -> tuple[np.ndarray, str | None, np.ndarray, str | None]:
        # Achieve an operation f on the objs

        fname = f.__name__

        compatible_units = fname in self.__compatible_units
        remove_units = fname in self.__remove_units
        quaternion_aware = fname in self.__quaternion_aware

        (
            fname,
            inputs,
            objtypes,
            returntype,
            is_masked,
            is_quaternion,
        ) = self._preprocess_op_inputs(fname, inputs)

        # Now we can proceed

        obj = cpy.copy(inputs.pop(0))
        objtype = objtypes.pop(0)

        other = None
        if inputs:
            other = cpy.copy(inputs.pop(0))
            othertype = objtypes.pop(0)

        # Is our first object a NDdataset
        # ------------------------------------------------------------------------------
        is_dataset = objtype == "NDDataset"

        # Get the underlying data: If one of the input is masked, we will work with
        # masked array
        d = obj._umasked(obj.data, obj.mask) if is_masked and is_dataset else obj.data

        # Do we have units?
        # We create a quantity q that will be used for unit calculations (without
        # dealing with the whole object)
        def reduce_(magnitude):
            if hasattr(magnitude, "dtype"):
                if magnitude.dtype in TYPE_COMPLEX:
                    magnitude = magnitude.real
                elif magnitude.dtype is typequaternion:
                    magnitude = as_float_array(magnitude)[..., 0]
                magnitude = magnitude.max()
            return magnitude

        q = reduce_(d)
        if hasattr(obj, "units") and obj.units is not None:
            q = Quantity(q, obj.units)
            q = q.values if hasattr(q, "values") else q  # case of nddataset, coord,

        # Now we analyse the other operands
        # ---------------------------------------------------------------------------
        args = []
        otherqs = []

        # If other is None, then it is a unary operation we can pass the following

        if other is not None:
            # First the units may require to be compatible, and if thet are sometimes
            # they may need to be rescales
            if (
                othertype in ["NDDataset", "Coord", "Quantity"]
                and not other.unitless
                and hasattr(obj, "units")
                and compatible_units
            ):
                # adapt the other units to that of object
                other.ito(obj.units)

            # If all inputs are datasets BUT coordset mismatch.
            if (
                is_dataset
                and (othertype == "NDDataset")
                and (other._coordset != obj._coordset)
            ):
                obc = obj.coordset
                otc = other.coordset

                # here we can have several situations:
                # -----------------------------------
                # One acceptable situation could be that we have a single value
                if other._squeeze_ndim == 0 or (
                    (obc is None or obc.is_empty) and (otc is None or otc.is_empty)
                ):
                    pass

                # Another acceptable situation is that the other NDDataset is 1D, with
                # compatible
                # coordinates in the x dimension
                elif other._squeeze_ndim >= 1:
                    try:
                        assert_coord_almost_equal(
                            obc[obj.dims[-1]],
                            otc[other.dims[-1]],
                            decimal=3,
                            data_only=True,
                        )  # we compare only data for this operation
                    except TypeError as err:
                        # This happen when coord are None or empty
                        xobc = (
                            None
                            if obc is None or obc[obj.dims[-1]].is_empty
                            else obc[obj.dims[-1]]
                        )
                        xotc = (
                            None
                            if otc is None or otc[other.dims[-1]].is_empty
                            else otc[other.dims[-1]]
                        )
                        # Warnning: copilot has changed obj by other.dims above. Check it
                        if xobc is None and xotc is None:
                            pass
                        else:
                            raise CoordinatesMismatchError(
                                obc[obj.dims[-1]].data,
                                otc[other.dims[-1]].data,
                            ) from err
                    except AssertionError as err:
                        raise CoordinatesMismatchError(
                            obc[obj.dims[-1]].data,
                            otc[other.dims[-1]].data,
                        ) from err

                # if other is multidimensional and as we are talking about element wise
                # operation, we assume
                # that all coordinates must match
                elif other._squeeze_ndim > 1:
                    for idx in range(obj.ndim):
                        try:
                            assert_coord_almost_equal(
                                obc[obj.dims[idx]],
                                otc[other.dims[idx]],
                                decimal=3,
                                data_only=True,
                            )  # we compare only data for this operation
                        except AssertionError as err:
                            raise CoordinatesMismatchError(
                                obc[obj.dims[idx]].data,
                                otc[other.dims[idx]].data,
                            ) from err

            if othertype in ["NDDataset", "Coord"]:
                # mask?
                if is_masked:
                    arg = other._umasked(other.data, other.mask)
                else:
                    arg = other.data

            else:
                # Not a NDArray.

                # if it is a quantity than separate units and magnitude
                arg = other.m if isinstance(other, Quantity) else other

            args.append(arg)

            otherq = reduce_(arg)

            if hasattr(other, "units") and other.units is not None:
                otherq = Quantity(otherq, other.units)
                otherq = (
                    otherq.values if hasattr(otherq, "values") else otherq
                )  # case of nddataset, coord,
            otherqs.append(otherq)

        # Calculate the resulting units (and their compatibility for such operation)
        # ------------------------------------------------------------------------------
        # Do the calculation with the units to find the final one

        def check_require_units(fname, _units):
            if fname in self.__require_units:
                requnits = self.__require_units[fname]
                if (
                    requnits in (DIMENSIONLESS, "radian", "degree")
                    and _units.dimensionless
                ):
                    # this is compatible:
                    _units = DIMENSIONLESS
                else:
                    if requnits == DIMENSIONLESS:
                        s = "DIMENSIONLESS input"
                    else:
                        s = f"`{requnits}` units"
                    raise DimensionalityError(
                        _units,
                        requnits,
                        extra_msg=f"\nFunction `{fname}` requires {s}",
                    )

            return _units

        # define an arbitrary quantity `q` on which to perform the units calculation

        units = UNITLESS

        if not remove_units:
            if hasattr(q, "units"):
                # q = q.m * check_require_units(fname, q.units)
                q = q.to(check_require_units(fname, q.units))

            for i, otherq in enumerate(otherqs[:]):
                if hasattr(otherq, "units"):
                    otherqm = otherq.m.data if np.ma.isMaskedArray(otherq) else otherq.m
                    otherqs[i] = otherqm * check_require_units(fname, otherq.units)
                elif fname in [
                    "add",
                    "sub",
                    "iadd",
                    "isub",
                    "and",
                    "xor",
                    "or",
                ] and hasattr(q, "units"):
                    otherqs[i] = otherq * q.units  # take the unit of the first obj

            # some functions are not handled by pint regardings units, try to solve this
            # here
            f_u = f
            if compatible_units:
                f_u = np.add  # take a similar function handled by pint

            try:
                res = f_u(q, *otherqs)

            except Exception as e:
                if not otherqs:
                    # in this case easy we take the units of the single argument except
                    # for some function where units
                    # can be dropped
                    res = q
                else:
                    raise e

            if hasattr(res, "units"):
                units = res.units

        # perform operation on magnitudes
        # ------------------------------------------------------------------------------
        if isufunc:
            with catch_warnings(record=True) as ws:
                # try to apply the ufunc
                if fname == "log1p":
                    fname = "log"
                    d = d + 1.0
                if fname in ["arccos", "arcsin", "arctanh"]:
                    if np.any(np.abs(d) > 1):
                        d = d.astype(np.complex128)
                elif fname in ["sqrt"] and np.any(d < 0):
                    d = d.astype(np.complex128)

                if fname == "sqrt":  # do not work with masked array
                    data = d ** (1.0 / 2.0)
                elif fname == "cbrt":
                    data = np.sign(d) * np.abs(d) ** (1.0 / 3.0)
                else:
                    data = getattr(np, fname)(d, *args)

                # if a warning occurs, let handle it with complex numbers or return an
                # exception:
                if ws and "invalid value encountered in " in ws[-1].message.args[0]:
                    ws = []  # clear
                    # this can happen with some function that do not work on some real
                    # values such as log(-1)
                    # then try to use complex
                    data = getattr(np, fname)(
                        d.astype(np.complex128),
                        *args,
                    )  # data = getattr(np.emath, fname)(d, *args)
                    if ws:
                        raise ValueError(ws[-1].message.args[0])
                elif ws and "overflow encountered" in ws[-1].message.args[0]:
                    warning_(ws[-1].message.args[0])
                elif ws:
                    raise ValueError(ws[-1].message.args[0])

            # TODO: check the complex nature of the result to return it

        else:
            # make a simple operation
            try:
                if (
                    not is_quaternion
                    or quaternion_aware
                    and all(arg.dtype not in TYPE_COMPLEX for arg in args)
                ):
                    data = f(d, *args)
                else:
                    # in this case we will work on both complex separately
                    dr, di = quat_as_complex_array(d)
                    datar = f(dr, *args)
                    datai = f(di, *args)
                    data = as_quaternion(datar, datai)

            except Exception as e:
                raise ArithmeticError(e.args[0]) from e

        # get possible mask
        if isinstance(data, np.ma.MaskedArray):
            mask = data._mask
            data = data._data
        else:
            mask = NOMASK  # np.zeros_like(data, dtype=bool)

        # return calculated data, units and mask
        return data, units, mask, returntype

    @_reduce_method
    @_from_numpy_method
    def amax(cls, dataset, dim=None, keepdims=False, **kwargs):
        """
        Return the maximum of the dataset or maxima along given dimensions.

        For quaternion data, the maximum is determined based on the real part.

        Parameters
        ----------
        dataset : array_like
            Input array or object that can be converted to an array.
        dim : None or int or dimension name or tuple of int or dimensions, optional
            Dimension or dimensions along which to operate.  By default, flattened input
            is used.
            If this is a tuple, the maximum is selected over multiple dimensions,
            instead of a single dimension or all the dimensions as before.
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the input array.

        Returns
        -------
        amax
            Maximum of the data. If `dim` is None, the result is a scalar value.
            If `dim` is given, the result is an array of dimension `ndim - 1` .

        See Also
        --------
        amin : The minimum value of a dataset along a given dimension, propagating NaNs.
        minimum : Element-wise minimum of two datasets, propagating any NaNs.
        maximum : Element-wise maximum of two datasets, propagating any NaNs.
        fmax : Element-wise maximum of two datasets, ignoring any NaNs.
        fmin : Element-wise minimum of two datasets, ignoring any NaNs.
        argmax : Return the indices or coordinates of the maximum values.
        argmin : Return the indices or coordinates of the minimum values.

        Notes
        -----
        For dataset with complex or hypercomplex type type, the default is the
        value with the maximum real part.
        """
        axis, dim = cls.get_axis(dim, allows_none=True)
        quaternion = False
        if dataset.dtype is typequaternion:
            quaternion = True
            data = dataset
            dataset = as_float_array(dataset)[..., 0]  # real part
        m = np.ma.max(dataset, axis=axis, keepdims=keepdims)
        if quaternion:
            if dim is None:
                # we return the corresponding quaternion value
                idx = np.ma.argmax(dataset)
                c = list(np.unravel_index(idx, dataset.shape))
                m = data[..., c[-2], c[-1]][()]
            else:
                m = np.ma.diag(data[np.ma.argmax(dataset, axis=axis)])

        if np.isscalar(m) or (m.size == 1 and not keepdims):
            if not np.isscalar(m):  # case of quaternion
                m = m[()]
            if cls.units is not None:
                return Quantity(m, cls.units)
            return m

        dims = cls.dims
        if hasattr(m, "mask"):
            cls._data = m.data
            cls._mask = m.mask
        else:
            cls._data = m

        # Here we must eventually reduce the corresponding coordinates
        if hasattr(cls, "coordset"):
            coordset = cls.coordset
            if coordset is not None:
                if dim is not None:
                    idx = coordset.names.index(dim)
                    if not keepdims:
                        del coordset.coords[idx]
                        dims.remove(dim)
                    else:
                        coordset.coords[idx].data = [
                            0,
                        ]
                else:
                    # find the coordinates
                    idx = np.ma.argmax(dataset)
                    c = list(np.unravel_index(idx, dataset.shape))

                    coord = {}
                    for i, item in enumerate(c[::-1]):
                        dim = dims[-(i + 1)]
                        id = coordset.names.index(dim)
                        coord[dim] = coordset.coords[id][item]
                    cls.set_coordset(coord)

        cls.dims = dims
        return cls

    @_reduce_method
    @_from_numpy_method
    def amin(cls, dataset, dim=None, keepdims=False, **kwargs):
        """
        Return the maximum of the dataset or maxima along given dimensions.

        For quaternion data, the minimum is determined based on the real part.

        Parameters
        ----------
        dataset : array_like
            Input array or object that can be converted to an array.
        dim : None or int or dimension name or tuple of int or dimensions, optional
            Dimension or dimensions along which to operate.  By default, flattened input
            is used.
            If this is a tuple, the minimum is selected over multiple dimensions,
            instead of a single dimension or all the dimensions as before.
        keepdims : bool, optional
            If this is set to True, the dimensions which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the input array.

        Returns
        -------
        amin
            Minimum of the data. If `dim` is None, the result is a scalar value.
            If `dim` is given, the result is an array of dimension `ndim - 1` .

        See Also
        --------
        amax : The maximum value of a dataset along a given dimension, propagating NaNs.
        minimum : Element-wise minimum of two datasets, propagating any NaNs.
        maximum : Element-wise maximum of two datasets, propagating any NaNs.
        fmax : Element-wise maximum of two datasets, ignoring any NaNs.
        fmin : Element-wise minimum of two datasets, ignoring any NaNs.
        argmax : Return the indices or coordinates of the maximum values.
        argmin : Return the indices or coordinates of the minimum values.
        """
        axis, dim = cls.get_axis(dim, allows_none=True)
        quaternion = False
        if dataset.dtype is typequaternion:
            quaternion = True
            data = dataset
            dataset = as_float_array(dataset)[..., 0]  # real part
        m = np.ma.min(dataset, axis=axis, keepdims=keepdims)
        if quaternion:
            if dim is None:
                # we return the corresponding quaternion value
                idx = np.ma.argmin(dataset)
                c = list(np.unravel_index(idx, dataset.shape))
                m = data[..., c[-2], c[-1]][()]
            else:
                m = np.ma.diag(data[np.ma.argmin(dataset, axis=axis)])

        if np.isscalar(m) or (m.size == 1 and not keepdims):
            if not np.isscalar(m):  # case of quaternion
                m = m[()]
            if cls.units is not None:
                return Quantity(m, cls.units)
            return m

        dims = cls.dims
        if hasattr(m, "mask"):
            cls._data = m.data
            cls._mask = m.mask
        else:
            cls._data = m

        # Here we must eventually reduce the corresponding coordinates
        if hasattr(cls, "coordset"):
            coordset = cls.coordset
            if coordset is not None:
                if dim is not None:
                    idx = coordset.names.index(dim)
                    if not keepdims:
                        del coordset.coords[idx]
                        dims.remove(dim)
                    else:
                        coordset.coords[idx].data = [
                            0,
                        ]
                else:
                    # find the coordinates
                    idx = np.ma.argmin(dataset)
                    c = list(np.unravel_index(idx, dataset.shape))

                    coord = {}
                    for i, item in enumerate(c[::-1]):
                        dim = dims[-(i + 1)]
                        id = coordset.names.index(dim)
                        coord[dim] = coordset.coords[id][item]
                    cls.set_coordset(coord)

        cls.dims = dims
        return cls


# Import parts of NDMath
# -----------------------------------------------------------------------------------
from spectrochempy.application.application import warning_
from spectrochempy.core.dataset.arraymixins.ndmath import _set_operators
from spectrochempy.core.dataset.arraymixins.ndmath import _update_api_funclist
from spectrochempy.core.units import DimensionalityError
from spectrochempy.core.units import Quantity
from spectrochempy.core.units import ur

# Import required modules/classes for NDMathQuaternion
# -----------------------------------------------------------------------------------
from spectrochempy.utils.constants import TYPE_COMPLEX
from spectrochempy.utils.orderedset import OrderedSet

# Set operators for NDMathQuaternion
# -----------------------------------------------------------------------------------
_set_operators(NDMathQuaternion, priority=50)

# Add module functions from NDMathQuaternion to the module
# -----------------------------------------------------------------------------------
api_funcs = _update_api_funclist(NDMathQuaternion)
thismodule = sys.modules[__name__]

for funcname in api_funcs:
    setattr(thismodule, funcname, getattr(NDMathQuaternion, funcname))
    thismodule.__all__.append(funcname)
