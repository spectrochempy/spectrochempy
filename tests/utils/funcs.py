ufunc_str = """

#Math operations

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
        unary = False
        if item[1].startswith('x, /['):
            unary = True
        ufuncs.append((item[0], unary ,item[1]))
    return ufuncs



if __name__ == '__main__':

    ufuncs = liste_ufunc()

    UNARY_MATH = [(a[0], a[2]) for a in liste_ufunc() if a[1]]
    print(UNARY_MATH)