.. _contributing_codebase:

=============================
Contributing to the code
=============================

.. contents:: Table of Contents:
   :local:

Code standards
--------------

During :ref:`Continuous Integration <contributing.ci>` testing, several
tools will be run to check your code for stylistic errors.
Generating any warnings will cause the test to fail.
Thus, good style is a requirement for submitting code to SpectroChemPy.

In addition, it is important that we
do not make sudden changes to the code that could have the potential to break
a lot of user code as a result, that is, we need it to be as *backwards compatible*
as possible to avoid mass breakages.

Some extra checks can be run by
``pre-commit`` - see :ref:`here <contributing.pre-commit>` for how to
run them.

Additional standards are outlined on the :ref:`spectrochempy code style guide <code_style>`.

.. _contributing.pre-commit:

Pre-commit
----------

We encourage you to use `pre-commit hooks <https://pre-commit.com/>`
to automatically run ``black``, ``flake8`` when you make a git commit. This
can be done by installing ``pre-commit``::

    pip install pre-commit


and then running::

    pre-commit install


from the root of the spectrochempy repository. Now all of the styling checks will be
run each time you commit changes without your needing to run each one manually.
In addition, using ``pre-commit`` will also allow you to more easily
remain up to date with our code checks as they change.

Note that if needed, you can skip these checks with ``git commit --no-verify``.

If you don't want to use ``pre-commit`` as part of your workflow, you can still use it
to run its checks with::

    pre-commit run --files <files you have modified>

without needing to have done ``pre-commit install`` beforehand.

If you want to run checks on all recently committed files on upstream/master you can use::

    pre-commit run --from-ref=upstream/master --to-ref=HEAD --all-files

without needing to have done ``pre-commit install`` beforehand.


Optional dependencies
---------------------

Optional dependencies (e.g., cantera, nmrglue, ...) should be imported with the private helper
``spectrochempy.optional.import_optional_dependency``. This ensures a
consistent error message when the dependency is not met.

All methods using an optional dependency should include a test asserting that an
``ImportError`` is raised when the optional dependency is not found. This test
should be skipped if the library is present.

All optional dependencies should be documented in
:ref:`install_adds` and the minimum required version should be
set in the ``spectrochempy.optional.VERSIONS`` dict.


.. _contributing.code-formatting:

Python (PEP8 / black)
~~~~~~~~~~~~~~~~~~~~~

spectrochempy follows the `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_ standard
and uses `Black <https://black.readthedocs.io/en/stable/>`_ and
`Flake8 <http://flake8.pycqa.org/en/latest/>`_ to ensure a consistent code
format throughout the project. We encourage you to use :ref:`pre-commit <contributing.pre-commit>`.

:ref:`Continuous Integration <contributing.ci>` will run those tools and
report any stylistic errors in your code. Therefore, it is helpful before
submitting code to run the check yourself::

   black spectrochempy
   git diff upstream/master -u -- "*.py" | flake8 --diff

to auto-format your code. Additionally, many editors have plugins that will
apply ``black`` as you edit files.

One caveat about ``git diff upstream/master -u -- "*.py" | flake8 --diff``: this
command will catch any stylistic errors in your changes specifically, but
beware it may not catch all of them. For example, if you delete the only
usage of an imported function, it is stylistically incorrect to import an
unused function. However, style-checking the diff will not catch this because
the actual import is not part of the diff. Thus, for completeness, you should
run this command, though it may take longer::

   git diff upstream/master --name-only -- "*.py" | xargs -r flake8

Note that on macOS, the ``-r`` flag is not available, so you have to omit it and
run this slightly modified command::

   git diff upstream/master --name-only -- "*.py" | xargs flake8

Windows does not support the ``xargs`` command (unless installed for example
via the `MinGW <http://www.mingw.org/>`__ toolchain), but one can imitate the
behavior as follows::

    for /f %i in ('git diff upstream/master --name-only -- "*.py"') do flake8 %i

This will get all the files being changed by the PR (and ending with ``.py``),
and run ``flake8`` on them, one after the other.

Note that these commands can be run analogously with ``black``.

Backwards compatibility
~~~~~~~~~~~~~~~~~~~~~~~

Please try to maintain backward compatibility. If you think breakage is required,
clearly state why as part of the pull request.  Also, be careful when changing method
signatures and add deprecation warnings where needed. Also, add the deprecated sphinx
directive to the deprecated functions or methods.

.. code-block:: python

    from spectrochempy.utils.exceptions import deprecated

    @deprecated("This function is deprecated and may be removed in future version")
    def old_func():
        """Summary of the function.

        .. deprecated:: 1.1.0
           Use new_func instead.
        """

        new_func()


    def new_func():
        pass


You'll also need to

1. Write a new test that asserts a warning is issued when calling with the deprecated argument
2. Update all of spectrochempy existing tests and code to use the new argument.

.. _contributing.ci:

Testing with continuous integration
-----------------------------------

The spectrochempy test suite will run automatically on `GitHub Actions <https://github.com/features/actions/>`,
once your pull request is submitted.

A pull-request will be considered for merging when you have an all 'green' build. If any tests are failing,
then you will get a red 'X', where you can click through to see the individual failed tests.


.. _contributing.tdd:

Test-driven development/code writing
------------------------------------

SpectroChemPy strongly encourages contributors to embrace
`test-driven development (TDD) <https://en.wikipedia.org/wiki/Test-driven_development>`_.
This development process "relies on the repetition of a very short development cycle:
first the developer writes an (initially failing) automated test case that defines a desired
improvement or new function, then produces the minimum amount of code to pass that test."
So, before actually writing any code, you should write your tests.  Often the test can be
taken from the original GitHub issue.  However, it is always worth considering additional
use cases and writing corresponding tests.

Adding tests is one of the most common requests after code is pushed to spectrochempy.  Therefore,
it is worth getting in the habit of writing tests ahead of time so this is never an issue.

Like many packages, spectrochempy uses `pytest
<https://docs.pytest.org/en/latest/>`_ and the convenient
extensions in `numpy.testing
<https://numpy.org/doc/stable/reference/routines.testing.html>`_.


Writing tests
~~~~~~~~~~~~~

All tests should go into the ``tests`` directory.
This folder contains many current examples of tests, and we suggest looking to these for
inspiration.  If your test requires working with files or
network connectivity, there is more information on the `testing page
<https://github.com/spectrochempy-dev/spectrochempy/wiki/Testing>`_ of the wiki.

The easiest way to verify that your code is correct is to
explicitly construct the result you expect, then compare the actual result to
the expected correct result.


Using ``pytest``
~~~~~~~~~~~~~~~~

Here is an example of a self-contained set of tests that illustrate multiple features that we like to use.

* functional style: tests are like ``test_*`` and *only* take arguments that are either fixtures or parameters
* ``pytest.mark`` can be used to set metadata on test functions, e.g. ``skip`` or ``xfail``.
* using ``parametrize``: allow testing of multiple cases
* to set a mark on a parameter, ``pytest.param(..., marks=...)`` syntax should be used
* ``fixture``, code for object construction, on a per-test basis
* using bare ``assert`` for scalars and truth-testing
* ``assert_dataset_equal`` for spectrochempy object comparisons.
* the typical pattern of constructing an ``expected`` and comparing versus the ``result``

We would name this file ``test_ds.py`` and put in an appropriate place in the ``tests/test_dataset`` structure.

See files in ``tests`` directory.

.. code-block:: python

    import numpy as np
    import pytest
    import spectrochempy as scp
    from spectrochempy.utils.testing import assert_array_equal


    def test_nddataset_real_imag():

        na = np.array(
            [[1.0 + 2.0j, 2.0 + 0j], [1.3 + 2.0j, 2.0 + 0.5j], [1.0 + 4.2j, 2.0 + 3j]])
        nd = scp.NDDataset(na)
        # in the last dimension
        assert_array_equal(nd.real, na.real)
        assert_array_equal(nd.imag, na.imag)


    adata = (
        [],
        [None, 1.0],
        [np.nan, np.inf],
        [0, 1, 2],
        [0.0, 1.0, 3.0],
        [0.0 + 1j, 10.0 + 3.0j],
        [0.0 + 1j, np.nan + 3.0j],
    )


    @pytest.mark.parametrize("a", adata)
    def test_1D_NDDataset(a):
        # 1D
        for arr in [a, np.array(a)]:
            ds = scp.NDDataset(arr)
            assert ds.size == len(arr)
            assert ds.shape == (ds.size,)
            if ds.size == 0:
                assert ds.dtype is None
                assert ds.dims == []
            else:
                assert ds.dtype in [np.float64, np.complex128]
                assert ds.dims == ["x"]
            # force dtype
            ds = scp.NDDataset(arr, dtype=np.float32)
            if ds.size == 0:
                assert ds.dtype is None
            else:
                assert ds.dtype == np.float32
            assert ds.title == "<untitled>"
            assert ds.mask == scp.NOMASK
            assert ds.meta == {}
            assert ds.name.startswith("NDDataset")
            assert ds.author == get_user_and_node()
            assert ds.description == ""
            assert ds.history == []

A test run of this using yields:

.. code-block:: shell

   ============= test session starts ======================
   platform darwin -- Python 3.8.8, pytest-6.2.2, py-1.10.0, pluggy-0.13.1
   rootdir: spectrochempy, configfile: pytest.ini
   plugins: flake8-1.0.7, anyio-2.2.0, doctestplus-0.9.0
   collected 8 items

   tests/test_dataset/test_ds.py .......  [100%]

   =========== warnings summary ============================
   tests/test_dataset/test_ds.py::test_1D_NDDataset[a5]
   tests/test_dataset/test_ds.py::test_1D_NDDataset[a6]
   spectrochempy/core/dataset/ndcomplex.py:152: ComplexWarning: Casting complex values to real discards the imaginary part
    data = data.astype(np.dtype(self._dtype), copy=False)

   -- Docs: https://docs.pytest.org/en/stable/warnings.html
   ============ 8 passed, 2 warnings in 0.28s ==============


Running the test suite
----------------------

The tests can then be run directly inside your Git clone by typing::

.. code-block:: bash

    pytest .

The test suite is exhaustive and takes several minutes to run.  Often it is
worth running only a subset of tests first around your changes before running the
entire suite.

The easiest way to do this is with::

.. code-block:: bash

    pytest spectrochempy/path/to/test.py -k regex_matching_test_name

Or with one of the following constructs::

.. code-block:: bash

    pytest tests/[test-module].py
    pytest tests/[test-module].py::[TestClass]
    pytest tests/[test-module].py::[TestClass]::[test_method]

Using `pytest-xdist <https://pypi.org/project/pytest-xdist>`_, one can
speed up local testing on multicore machines. To use this feature, you will
need to install ``pytest-xdist`` via::

.. code-block:: bash

    mamba install pytest-xdist

This can significantly reduce the time it takes to locally run tests before
submitting a pull request.

For more, see the `pytest <https://docs.pytest.org/en/latest/>`_ documentation.


Documenting change log
-----------------------

Changes should be reflected in the release notes located in ``CHANGELOG.md`` in the root directory of the spectrochempy package.
This file contains an ongoing change log for each release.  Add an entry to this file to
document your fix, enhancement or (unavoidable) breaking change.  Include the
GitHub issue number when adding your entry (using ``(issue #1234)``` where ``1234`` is the
issue/pull request number).
