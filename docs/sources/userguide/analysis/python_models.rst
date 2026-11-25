Python models for one-dimensional fitting
=========================================

``Model`` and additive profiles feed the same canonical definition and fitting
infrastructure as the ``Optimize`` script DSL. The DSL remains supported.
This entry point is for scalar-parameter, real 1D functions; the existing
``peaks=...`` and full-grid ``model=callback, parameters=...`` entries retain
their separate ND contracts.

Two linked peaks and a background
---------------------------------

.. code-block:: python

    model = (
        scp.Gaussian(prefix="p1_")
        + scp.Gaussian(prefix="p2_")
        + scp.Linear(prefix="bg_")
    )
    params = model.make_params(p1_center=1600, p2_center=1630)
    params["p1_width"].set(value=10, min=1, max=30)
    params["p2_width"].set(expr="p1_width")
    params["p1_center"].set(vary=False)
    opt = scp.Optimize(model=model, params=params, method="least_squares").fit(ds)
    _ = opt.plot()
    print(opt.report())
    print(opt.result.fitted_parameters)

``Gaussian`` uses public names ``amplitude``, ``center`` and ``width``. They
map to the existing ``gaussianmodel`` parameters ``ampl``, ``pos`` and ``width``.
Its numerical formula is exactly the existing normalized ``scp.gaussian``:
sigma is ``width / 2.3548`` and the Gaussian density is multiplied by the
absolute difference between the first two coordinate samples. On a sufficiently
extended uniform grid the sample sum approximates ``amplitude``. This parameter
is neither peak height nor a grid-independent integral. Changing grid spacing
changes the profile values at a fixed amplitude. No new normalization is implied.
Width is FWHM in coordinate magnitudes; a varying width requires a strictly
positive lower bound (default ``1e-12``). A fixed or linked width must also
remain positive. Center uses coordinate magnitudes. The existing 1D coordinate
handling, including its precision conventions, is preserved.

``Linear`` is ``slope*x + intercept`` on absolute coordinates: intercept has
intensity magnitudes, slope intensity per coordinate magnitude. Unlike the
existing polynomial baseline, it does not center x. It is an explicitly fitted
additive component, with no automatic subtraction or intensity rescaling.

A personal function
-------------------

.. code-block:: python

    def decay(x, amplitude, rate=0.5, offset=0.0):
        return amplitude * np.exp(-rate*x) + offset

    model = scp.Model(decay, prefix="custom_")
    params = model.make_params(custom_amplitude=2.0)
    params["custom_rate"].set(min=0.001, max=10)
    opt = scp.Optimize(model=model, params=params).fit(ds)

The function receives the numerical coordinate vector followed by positional
scalar arguments. It must return a finite real vector on that grid and should
be deterministic, without mutating external state. There is no implicit peak
normalization. Scalar defaults supply initial values; required arguments need
``make_params`` overrides. Variadic and keyword-only arguments are rejected.
Input values are finite real magnitudes; this facade does not convert Pint
quantities or validate dimensions of expressions.

Names, parameters and links
---------------------------

Parameter names are lowercase Python identifiers. A prefix is empty or a
lowercase identifier ending in ``_``; it is prepended to each argument name.
Component prefixes must be unique, including the empty prefix. Addition
preserves component order and rejects duplicate prefixes or full-name
collisions. Reusing a Model or a Parameter object never creates sharing.

``make_params`` accepts scalar or ``Parameter`` overrides for known model names.
Scalar overrides retain the model's default bounds. A ``Parameter`` override
replaces its entire declaration. Unknown or missing required names are errors.
``Parameters.add`` can add an auxiliary scalar used by a link; unused
auxiliaries are rejected. Assignment and ``copy()`` copy declarations.

``Parameter.set(value=..., min=..., max=..., vary=...)`` updates initial
configuration atomically. Bounds are inclusive and must satisfy min < max;
use ``vary=False`` to fix a value, not equal bounds. Initial independent values
must lie within their bounds. None removes a bound.
For a free parameter the initial value must be strictly inside any finite
bounds. The current 1D bounded transform can have zero derivative exactly at an
endpoint and falsely stop there; the facade rejects that initialization rather
than silently moving it. Fixed values may equal a bound. If a previous fit ends
exactly on a bound, create another estimator with an interior starting value
before refitting. Improving that transform is a separate engine change.

``set(expr="p1_width")`` explicitly links a parameter. Expressions accept full
names, finite numerical literals, parentheses, unary signs, and ``+ - * / **``.
They may reference free, fixed or other linked parameters, including forward
references. Unknown names, cycles, calls, attributes, indexing and nonfinite or
undefined initial expressions are rejected before fitting. A trial expression
or model outside its domain fails rather than being clipped silently.
Math functions are intentionally not part of this first expression language;
this restriction applies only to the new Python facade, not the historical DSL.

A link replaces the target's independent bounds and makes it nonvarying.
Bounds on a linked parameter are not supported: bound its source instead.
Supplying bounds together with ``expr`` is an error. ``set(expr=None)`` removes
the link and makes the parameter free unless ``vary`` is also supplied.
For example, a positive auxiliary difference can define an ordered center:

.. code-block:: python

    params.add("separation", value=30, min=1, max=100)
    params["p2_center"].set(expr="p1_center + separation")

Configuration and results
-------------------------

``Optimize`` copies the model topology and parameter declarations on entry.
Changing the supplied configuration later does not affect the estimator or
another fit. ``opt.initial_params`` returns a copy of that initial configuration.
The existing ``opt.params()`` method keeps its estimator-inspection meaning.
Repeated fitting follows the existing 1D behavior: the last solution is the
starting point, while ``initial_params`` stays unchanged. Create another
estimator to start again from the original declarations.

If all parameters are fixed or depend only on fixed values, an ordinary fit
evaluates the model and residuals without invoking a solver. It reports
``optimization_performed=False`` with an explanatory message; success means
that evaluation completed, not that parameters were optimized. No free-parameter
Jacobian, covariance or stderr is invented. This remains distinct from
``dry=True``, which retains the existing starting-model preview behavior.

Use the canonical ``opt.result`` for scientific outputs:

* ``fitted_parameters`` maps public names to physical fitted magnitudes,
  including fixed and linked values.
* ``parameter_names`` identifies the free parameters in covariance, stderr
  and confidence-interval order. ``opt.jacobian`` retains the raw residual
  Jacobian with respect to the optimizer coordinates, including the historical
  bound transformation; its columns follow that same order.
* ``parameters`` remains the estimator run configuration, not a solution.
* ``fitted``, ``components``, ``residuals`` and diagnostics retain their
  existing meanings. Component rows follow addition order; internal component
  labels and generated script names are implementation details.

Uncertainty is a local free-parameter approximation in physical magnitudes.
The bound-transform derivatives are applied to the solver covariance; the raw
``opt.jacobian`` remains unchanged for compatibility. This applies equally to
Python and COMMON models. The approximation requires convergence, positive
residual degrees of freedom and a full-rank Jacobian. Covariance, stderr,
correlation and confidence intervals are unavailable when these conditions
fail, or when a fitted parameter is at or near a bound.
``diagnostics['covariance_reason']`` gives the exclusion reason, or ``None``
when available; ``covariance_coordinates='physical'`` and
``jacobian_coordinates='optimizer'`` identify the two coordinate systems.
The bound neighborhood is ``sqrt(machine epsilon)`` times the interval width
for two finite bounds, or times the maximum of one, the parameter magnitude
and the bound magnitude for one finite bound. This is a conservative
uncertainty qualification, not a test of physical stationarity. Historical
solver success alone can still include an erroneous stop at a bound; the
strictly interior starting-value requirement remains in place.

Propagation to linked/derived parameters is not implemented;
``diagnostics['derived_uncertainty_status']`` explicitly reports
``'not_propagated'``. Fixed or linked parameters receive no invented stderr.
``report()`` returns text based on these outputs; ``plot()`` forwards to the
existing ``plot_merit()`` view. Neither introduces a second result object.

Historical options and scope
----------------------------

The new entry requires ``autobase=False``, ``autoampl=False`` and
``amplitude_mode='area'`` (its defaults). Here ``area`` selects raw function
evaluation with the existing profile convention; it does not change a custom
function's physical meaning. ``height`` is rejected because the historical
global height normalization would rescale personal functions and linear
backgrounds. Automatic baseline/scale estimation and amplitude guessing are
also rejected. Existing DSL defaults and behavior remain unchanged.

``model=Model(...)`` cannot be combined with ``script``, ``peaks``,
``shared_parameters``, ``parameters``, ``usermodels``, ``linear_amplitudes``,
``solver_options`` or ``warm_start=True``.
The legacy ``constraints`` container is also excluded; declare bounds and links
in ``params`` instead. ``params`` requires a Model object;
grid callbacks continue to use ``parameters``. Replace an estimator explicitly
to switch from Python topology to a script. Custom Python functions are not
serialized by a rendered script.

``scale='log'`` is not exposed as an optimization capability: the 1D backend
currently implements bounded physical parameters, not logarithmic coordinates.
Requests through ``Parameter.set`` or ``Parameters.add`` raise
``NotImplementedError``. A separate engine increment must define transforms,
physical Jacobians and uncertainties and qualify performance before exposing it.
Units and derived uncertainty propagation likewise require separate work.
The reusable parameter facade does not specify a kinetics model, multi-experiment
input or a change to MCRALS.

See the executable :ref:`sphx_glr_gettingstarted_examples_gallery_auto_examples_analysis_c_curvefitting_plot_python_models.py`
example for a complete generated dataset and fit.
