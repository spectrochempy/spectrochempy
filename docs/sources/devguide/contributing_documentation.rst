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
- Provide high-level overviews and examples
- All code examples are tested

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
