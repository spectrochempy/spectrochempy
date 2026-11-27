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
Parameter declarations accept inclusive endpoint values. In the default
``bounds_mode='transform'``, Optimize requires free initial values strictly
inside finite bounds. The opt-in native mode below accepts endpoints. The
historical transform can have zero derivative exactly at an endpoint and
falsely stop there; transform mode rejects such Python initialization. Fixed
values may equal a bound. If a transform-mode fit ends exactly on a bound,
create another estimator with an interior start or select native mode before
refitting.

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
  Jacobian; its columns follow that same order. Transform mode uses optimizer
  coordinates, native mode physical magnitudes.
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
``jacobian_coordinates='optimizer'`` identify the two coordinate systems in
transform mode; native mode uses physical coordinates for both.
The bound neighborhood is ``sqrt(machine epsilon)`` times the interval width
for two finite bounds, or times the maximum of one, the parameter magnitude
and the bound magnitude for one finite bound. This is a conservative
uncertainty qualification, not a test of physical stationarity. Historical
solver success alone can still include an erroneous stop at a bound; the
strictly interior starting-value requirement remains in place in transform mode.

Propagation to linked/derived parameters is not implemented;
``diagnostics['derived_uncertainty_status']`` explicitly reports
``'not_propagated'``. Fixed or linked parameters receive no invented stderr.
``report()`` returns text based on these outputs; ``plot()`` forwards to the
existing ``plot_merit()`` view. Neither introduces a second result object.

Physical stationarity qualification
-----------------------------------

For 1D least-squares Python and COMMON fits, ``result.diagnostics`` also
reports a local physical-coordinate check, independently of the solver's
``success``, ``status`` and ``message``:

* ``physical_stationarity_status`` is ``'satisfied'``, ``'not_satisfied'``,
  ``'unavailable'`` or ``'not_checked'``.
* ``physical_projected_gradient`` is the largest absolute residual/Jacobian
  column cosine after projection onto feasible directions. A positive gradient
  exactly on a lower bound or a negative gradient exactly on an upper bound
  is projected out. A nearby strictly interior value retains both directions.
  The score is dimensionless; ``physical_stationarity_tolerance`` is ``1e-6``.
* ``physical_stationarity_reason`` explains an unavailable check, or reports
  ``'residual_at_roundoff'`` when every residual is at most 32 machine epsilons
  times the largest finite unmasked observed magnitude.
* ``stationarity_evaluations`` counts additional model calls used by this
  diagnostic. Regular interior fits use the retained Jacobian and signed
  transform derivatives, without extra model evaluations.

Near bounds, the check uses independent feasible physical probes because
the optimizer-coordinate Jacobian can lose the physical gradient there.
Three-point derivatives at two step sizes estimate derivative and roundoff
errors. Their estimated effect on the normalized gradient must be below a
quarter of the stationarity tolerance, and the resulting score interval must
lie entirely on one side of that tolerance. Otherwise the check is unavailable.
An ordinary one-sided check reuses
one probe and needs three model calls per column; a centered check needs four.
Probes evaluate copies of the fitted definition, including its parameter
links. Failed, nonfinite, flat or unresolved derivatives make the check
unavailable. Personal functions must remain deterministic.

The neighborhood triggering probes matches the conservative covariance
qualification above; it does not establish bound activity for projection.
Its one-sided absolute scale is expressed in the configured numerical
magnitudes; this is not unit-aware validation. This numerical first-order
check does not establish a global minimum or parameter identifiability.
An approximate boundary solution that is still strictly interior can fail
this check despite negligible remaining improvement. Finite probes are local
numerical estimates; interior columns retain the backend Jacobian approximation.
It does not change the solution, solver success, covariance policy or starting
values, and it does not retry an optimization. In particular, a historical
COMMON fit starting at an endpoint may still report solver success together
with ``physical_stationarity_status='not_satisfied'``. The strictly interior
starting-value requirement remains in place in the default transform mode.
Dry previews and all-fixed evaluations report ``'not_checked'``; solvers
without a retained least-squares Jacobian report ``'unavailable'``.
The existing ND diagnostics are unchanged.

Native bounds for one-dimensional least squares
-----------------------------------------------

``bounds_mode='native'`` is an explicit alternative for COMMON and composable
Python 1D fits using ``method='least_squares'`` or its ``'leastsq'`` alias.
The default ``'transform'`` preserves the historical engine and requires
strictly interior free starts in the Python facade. Parameter declarations
now validate their inclusive physical bounds; the estimator validates whether
its selected policy accepts an endpoint. A fixed parameter may equal a bound.

For example, with a 1D dataset ``ds``:

.. code-block:: python

    model = scp.Linear()
    params = model.make_params(
        slope=scp.Parameter(0, min=0),
        intercept=scp.Parameter(0, vary=False),
    )
    opt = scp.Optimize(
        model=model, params=params, bounds_mode="native"
    ).fit(ds)
    print(opt.result.fitted_parameters)
    print(opt.result.diagnostics)

Native mode optimizes physical free magnitudes with bounded TRF, three-point
finite differences and ``x_scale='jac'``. It sets ``ftol=None``, ``gtol=None``
and ``xtol=1e-12``: cost-change and absolute scaled-gradient stopping are disabled
because either can produce premature stops after endpoint initialization,
including small-signal fits. Relative step-size stopping remains enabled so
finite-difference gradient noise need not consume the entire evaluation budget
near an optimum. The evaluation budget is
``max_fun_calls`` or ``4 * max_iter``; exhausting it reports solver failure.
Three-point differences and physical stationarity probes require additional
model calls beyond the solver's ``nfev`` counter. Native mode may cost more
than the historical route and does not restart an optimization automatically.

The supplied initial configuration remains unchanged. As documented by
`SciPy least_squares <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html>`_,
TRF can adjust an endpoint internally to obtain a strictly feasible iterate;
this is not an application-level physical epsilon or a different declaration.
An exactly fitted, non-identifiable model can still exhaust the evaluation
budget with this step-only policy. Its zero residual does not turn solver
failure into success; inspect the physical stationarity and rank diagnostics.
The backend suppresses only SciPy trust-region division warnings when the final
residual is exactly zero. Warnings from the model remain visible. An early stop
at the diagnostic roundoff floor is not part of this policy.

The local relative step-size criterion does not guarantee physical stationarity,
identifiability or a global optimum. Read the independent physical diagnostic
as well as solver success. In particular, an approximate constrained solution
can remain strictly interior: the physical check then retains both directions
and can report ``'not_satisfied'`` despite negligible remaining improvement.
A nearly exact fit can also retain a normalized gradient above its threshold
when its remaining residual is above the diagnostic roundoff floor. Solver
tolerances and the physical cosine diagnostic are different criteria.

``result.parameters['bounds_mode']`` records the selected policy. Native
optimizations also record ``solver_method='trf'``, ``termination_policy='step'``,
``solver_gtol``, ``solver_xtol``, ``solver_optimality``, ``solver_nfev`` and
``solver_max_nfev``. Their raw Jacobian is
already physical, labelled ``jacobian_coordinates='physical'``; covariance
uses the identity coordinate transform and the existing physical free order.
The default engine retains its raw optimizer-coordinate Jacobian. Neither
mode changes the existing exclusions for unavailable uncertainty, including
near-bound solutions. Linked uncertainty propagation remains unavailable.

Native mode rejects simplex, basinhopping, legacy ``constraints``, ND peaks,
full-grid callbacks and named recovery/decay laws. Those routes keep their
existing policies. The existing ``solver_options`` mapping belongs to grid
callbacks and is not a native 1D options surface. Width positivity remains a
model-domain requirement, including ``min > 0`` for varying Gaussian widths.
This increment adds no open/closed-bound syntax, logarithmic coordinates or
unit-aware expressions. A new estimator is still needed to change topology;
initial/fitted ownership and the restrictions on historical automatic options
for composed Python models remain unchanged.

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
