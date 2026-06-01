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
- Written in reST or Jupyter notebooks
- Located in ``docs/userguide/``
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

On the published site, the root of ``gh-pages`` is the ``latest``
documentation. Stable releases are published under version directories such as
``0.9.2/``.

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
