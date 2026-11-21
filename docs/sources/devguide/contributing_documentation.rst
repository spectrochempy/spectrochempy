.. _contributing_documentation:

Contributing to Documentation
=============================

Documentation is crucial for SpectroChemPy. You don't need to be an expert to help - if something isn't clear, improving it helps everyone!

.. contents:: Contents
   :local:
   :depth: 2

Documentation Structure
-----------------------

SpectroChemPy documentation consists of:

1. **Docstrings** in the code (API reference)
2. **Guides and tutorials** in the ``docs/`` folder
3. **Examples** in Jupyter notebooks

We use:

- **reStructuredText** (reST) markup
- **Sphinx** documentation builder
- **NumPy docstring standard**
- **Jupyter notebooks** for tutorials

Documentation Sources
---------------------

API Documentation
~~~~~~~~~~~~~~~~~

- Lives in docstrings within Python code
- Follows :ref:`NumPy docstring standard <docstring>`
- Examples are tested during builds
- Located in ``docs/reference``

Guides & Tutorials
~~~~~~~~~~~~~~~~~~
- Written in reST or Jupytext Python files with ``# %%`` cells
- Located in ``docs/sources/userguide/``
- Edit the Jupytext ``.py`` source; the build generates and executes its
  paired ``.ipynb`` file. Do not commit generated notebooks.
- Provide high-level overviews and examples
- All code examples are tested

Gallery Examples
~~~~~~~~~~~~~~~~

Add gallery scripts under ``src/spectrochempy/examples/<category>/`` and
use the ``plot_<topic>.py`` naming convention. Keep their input datasets
in ``docs/data/``, alongside the data used by tutorial notebooks, rather
than beside the scripts. Plugin examples stay in their plugin's examples
directory. Use the preview commands below to execute either kind of page.

Building Documentation
----------------------

Quick Start
~~~~~~~~~~~
1. Set up development environment (:ref:`guide <contributing.environment>`)
2. From the repository root, build HTML::

    python docs/make.py html

3. View at ``build/html/index.html``

For documentation-only edits where executing examples is not needed, use::

    python docs/make.py --no-exec html

On the published site, the root contains the latest stable documentation.
Unreleased documentation from ``master`` is published under ``latest/``.
Stable releases also remain available under version directories such as
``VERSION/``. The stable snapshot is built from its published Git tag and is
never replaced with documentation built from ``master``. Documentation-focused
branches publish identifiable previews under their sanitized branch name and do
not redirect to stable documentation.

Build Options
~~~~~~~~~~~~~

- Full rebuild::

    python docs/make.py clean
    python docs/make.py html

- Build a single file::

    python docs/make.py --single-doc <path-to-file-relative-to-docs>.rst
    python docs/make.py --single-doc <path-to-file-relative-to-docs>.ipynb

- Build a single directory::

    python docs/make.py --directory <path-to-directory-relative-to-docs>

- Build a single API entry::

    python docs/make.py --single-doc spectrochempy.<class-or-method-name>
    # where ``<class-or-method-name>`` is the name of an importable method in the API.

Preview a Gallery Example or Tutorial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From the repository root, use the documentation environment to render the
page you are editing. For a gallery example, pass the script path relative to
the repository root. This also works with scripts under a plugin's
``examples/`` directory.

.. code-block:: bash

    python -m docs.make html --gallery-example src/spectrochempy/examples/core/a_nddataset/plot_a_create_dataset.py

For a tutorial, pass its Jupytext source path relative to ``docs/sources/``.
The preview reads the Python file directly.

.. code-block:: bash

    python -m docs.make html --single-doc userguide/plotting/plot_types.py

Each command prints the absolute path to the generated HTML page under
``build/preview/``. Open that page in a browser and keep the surrounding
files so its styles and figures load. The preview uses the data and plugins
available in the local documentation environment. Tutorials also need a
working Jupyter ``python3`` kernel. For prose or layout edits, add
``--no-exec`` to render the page without running its code or regenerating its
figures.

If your browser or editor preview needs an HTTP URL, reuse a local server
already serving this worktree's ``build/preview/`` directory. To start one,
run this command from the repository root.

.. code-block:: bash

    python -m http.server 8765 --bind 127.0.0.1 --directory build/preview

If port 8765 is busy, choose a free port and use it in the preview URL.
Append the printed HTML path relative to ``build/preview/`` to
``http://127.0.0.1:8765/``. After editing, rerun the preview command and
refresh the page in your browser. The URL works while the server is running.

Documentation Data
~~~~~~~~~~~~~~~~~~

Public input data for the tutorials and galleries is versioned in
``docs/data/``. The normal build verifies its size and SHA-256 manifest,
then selects this directory for Sphinx examples and notebook kernels.
It does not download or extract the full ``spectrochempy_data`` repository.
These files are excluded from wheel and source distributions. Corn remains
separate, as described below.

Adding or Updating Input Files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Put only the files needed by the example or notebook in
   ``docs/data/<data-family>/<dataset>/``. When copying an existing fixture,
   retain its path relative to the original ``testdata/`` directory.
2. Preserve the original file bytes and attribution notices. For a new
   dataset, add a README recording the source URL or DOI, version, authors
   and license. Keep private data such as Corn outside this directory.
3. Update ``docs/data/manifest.json`` with the relative path, byte count
   and SHA-256 for each added or changed file, including dataset notices.
   Keep its source revision accurate; document additions from other sources
   in their dataset README. Do not include generated plots or exported data.
4. Run the affected gallery or tutorial preview, then the full documentation
   build to check that the selection is sufficient. A missing or changed
   input fails verification instead of triggering a full-corpus download.

For example, inspect a fixture's manifest values from the repository root:

.. code-block:: python

    import hashlib
    from pathlib import Path

    data = Path("docs/data")
    path = data / "irdata/nh4y-activation.spg"
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    print(path.relative_to(data).as_posix(), path.stat().st_size, digest)

See ``docs/data/README.md`` for the current selection's provenance.

Reading Data in Examples and Notebooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use paths relative to ``scp.preferences.datadir`` in both gallery scripts
and Jupytext notebooks. The documentation builder sets this preference to
the checkout's ``docs/data/`` before executing each page. The reader resolves
its relative input there; it does not depend on the script or notebook's
working directory.

.. code-block:: python

    import spectrochempy as scp

    dataset = scp.read("irdata/nh4y-activation.spg", local_only=True)

For a format-specific reader, use the same path and option, for example
``scp.read_omnic("irdata/nh4y-activation.spg", local_only=True)``. For NumPy
or ordinary file I/O, obtain the full path from the preference:

.. code-block:: python

    filename = scp.preferences.datadir / "ramandata/hoyer_hess_2024/Fig_2-a-c_raw-data.txt"
    with filename.open("rb") as stream:
        first_line = stream.readline()

Do not hardcode ``docs/data/``, a home directory or ``../`` paths in the
example itself. Use ``local_only=True`` for local-only reads and examples
of missing files, so the reader cannot try a remote fallback. Keep data
retrieval out of the executed documentation path.

Outside a documentation build, the preference points to the user's usual
data directory; installing SpectroChemPy does not install ``docs/data/``.
State how a standalone reader can obtain any new dataset. Existing Raman
scripts use the local fixture when present and fetch its public archive
only when run without it. Corn uses the separate environment variables below.
For a one-off run from a checkout, prefer the gallery or tutorial preview
command: it selects the bundled data and isolates saved preferences. Setting
``scp.preferences.datadir`` manually persists in the user's configuration.

Documentation CI Artifacts
~~~~~~~~~~~~~~~~~~~~~~~~~~

Trusted upstream master/tag builds use a private copy of the Eigenvector
Corn dataset. Its source, size and SHA-256 are pinned in
``.github/data/corn.json``. The read-only deploy key
``SCPY_CORN_DATA_SSH_KEY`` grants access only to the private
``spectrochempy/spectrochempy-ci-data`` repository; the repository variable
``SCPY_CORN_DATA_REVISION`` pins its full commit SHA. Updating that revision
or the checksum is an explicit maintainer operation.

The staging action verifies the file before placing the raw MAT file and a
ZIP containing it in the runner's temporary directory. It exports
``SCPY_CORN_DATA`` and ``SCPY_CORN_ZIP`` for the examples, then removes the
private checkout. The working copies are removed after execution and are
outside the public test-data cache and documentation artifact inputs.
Do not add these files to public caches, artifacts or data repositories.

Pull request jobs and fork workflows do not run the private-data action and
retain the public Eigenvector download. Local users may also point those
two environment variables to their own Corn files. The scientific examples
and readers otherwise use the same data and processing. External Read the
Docs previews continue to use the public source, without a private key.

Production uses native RTD execution: notebooks are preexecuted on RTD before
the HTML render. The documentation artifact workflow is disabled in GitHub,
and its source only accepts manual dispatch for future experiments. It is
not a prerequisite for publishing the documentation. If explicitly re-enabled,
its tutorial and gallery jobs check out the workflow commit.
The tutorial job saves executed notebooks. The gallery job saves generated
reStructuredText, figures, thumbnails, downloads, and API backreferences.
The final job checks the commit and file hashes of both artifacts before
rendering one Sphinx site with the usual configuration and document paths.
Download the HTML artifact from the workflow run to inspect the result.

Use the pinned Sphinx-Gallery version from the documentation dependencies.
Version 0.22.1 includes portable backreference filenames: local Python classes
may have ``<locals>`` in their qualified names, but ``<`` and ``>`` are rejected
by GitHub's artifact uploader. The qualified names remain available in the
backreference JSON, including the API mini-galleries. The source artifact
format and its commit, package-version and file-hash checks are unchanged.

Before enabling artifact mode in production, validate it on a separate Read
the Docs test project with the same build time limit. Configure
``SCPY_RTD_DOCS_ARTIFACTS=1`` and set ``SCPY_RTD_GITHUB_TOKEN`` as a private
environment variable with read access to Actions in the SpectroChemPy
repository. Verify that the hosted build uses artifacts from its exact
checkout SHA and installed SpectroChemPy version, keeps the expected URLs and navigation, and completes within
the time limit, including dependency installation, the CI wait, and rendering.
If the concurrent workflow and RTD build exceed that limit, arrange for CI to
finish before triggering RTD. External pull request previews execute their
sources inside RTD.

RTD waits up to 15 minutes for a successful workflow with both artifacts for
its checkout. If artifacts expire, start a new workflow run with
``workflow_dispatch`` in the upstream SpectroChemPy repository. Verify that
the run uses the exact checkout SHA and retry RTD once CI succeeds. If the
ref has moved and you need the old SHA, dispatch CI on a temporary branch at that SHA and build
the corresponding RTD version of the temporary branch. To retry the original
RTD version instead, generate artifacts for its current SHA. To restore
execution inside RTD, unset ``SCPY_RTD_DOCS_ARTIFACTS``.

An immutable release checkout keeps its own dependency pins. Re-running the
artifact workflow for an older release does not apply a dependency fix from
``master``. In particular, the 1.4.0 checkout pins Sphinx-Gallery 0.19.0 and
must keep native RTD execution while no compatible artifact pair exists for
its exact commit and package version. Do not move its Git tag or substitute
artifacts from a newer commit. Keep the project-wide artifact flag unset
until every affected RTD version has a qualified artifact path; use a separate
test project to qualify the corrected checkout first.

If all artifact-production runs for the checkout have completed and the
most recent run failed, RTD stops with a link to that run rather than
waiting for the full timeout. An active retry can still complete, and an
existing usable pair from a successful run for the same SHA is preferred.

The documentation commands emit ``[docs-build]`` start/end markers and
elapsed times for dependency imports, gallery execution, artifact lookup,
download/extraction, manifest verification and HTML rendering. Exceptions
are also written to the original standard-error stream so gallery output
capture does not hide their traceback. During command execution, Python
thread stacks are dumped every two minutes until the command finishes.
Use the last phase marker and stacks to distinguish a slow or blocked
operation from a missing artifact. These diagnostics do not establish the
cause of a timeout on their own.

Writing Tips
------------

1. Use active voice
2. Be concise
3. Include practical examples
4. Test all code examples
5. Link related sections
6. Follow :ref:`docstring guide <docstring>`

See the `Sphinx reStructuredText primer <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_
for detailed syntax guide.
