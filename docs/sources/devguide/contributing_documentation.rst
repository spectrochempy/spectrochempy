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
2. Navigate to ``docs/`` directory
3. Build HTML::

    python make.py html

4. View at ``docs/build/html/latest/index.html``

Build Options
~~~~~~~~~~~~~

- Full rebuild::

    python make.py clean
    python make.py html

- Build a single file::

    python make.py --single-doc <path-to-file-relative-to-docs>.rst
    python make.py --single-doc <path-to-file-relative-to-docs>.ipynb

- Build a single directory::

    python make.py --directory <path-to-directory-relative-to-docs>

- Build a single API entry::

    python make.py --single-doc spectrochempy.<class-or-method-name>
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
