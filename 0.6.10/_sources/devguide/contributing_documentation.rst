.. _contributing_documentation:

=================================
Contributing to the documentation
=================================

Contributing to the documentation is very important for the SpectroChemPy project.
We encourage you to help us improve the documentation, and you don't need to be a SpectroChemPy expert to do so!
If something in the docs doesn't make sense to you, updating the the relevant section, after you understand,
is a great way to make sure it will help someone else.

.. contents:: Documentation:
   :local:


About the spectrochempy documentation
-------------------------------------

The documentation is written in **reStructuredText**, which is almost like writing
in plain English, and built using `Sphinx <https://www.sphinx-doc.org/en/master/>`__ . The
Sphinx Documentation has an excellent `introduction to reST
<https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`__ . Review the Sphinx docs to perform more
complex changes to the documentation as well.

It worth to note that:

* The spectrochempy documentation consists of two parts: the docstrings in the code itself
  and the content of the ``docs/`` folder.

  The docstrings provide an explanation of the usage of the individual
  functions, while the documentation in the ``docs`` folder consists of tutorial-like
  overviews per topic together with some other information (what's new,
  installation, etc).

* The docstrings follow the **Numpy Docstring Standard**.

  Follow the :ref:`spectrochempy docstring guide <docstring>` for detailed
  instructions on how to write a correct docstring.

  .. toctree::
     :maxdepth: 2

     contributing_docstring.rst

* The documentation, examples and tutorials make an heavy use of the `Jupyter Notebook <https://https://jupyter.org>`_.

  Almost all code examples in the docs and docstrings examples in the code itself are run during the
  doc build. This approach means that code examples will always be up to date.

* Our API documentation files in ``docs/userguide/reference`` house the auto-generated
  documentation from the docstrings.

Updating a spectrochempy docstring
----------------------------------

When improving a single function or method's docstring, it is not necessarily
needed to build the full documentation (see next section).
Indeed, one can use a script (adapted from Pandas) that checks a docstring (for example for the ``spectrochempy.round`` method)::

    python scripts/validate_docstrings.py spectrochempy.round

This script will indicate some formatting errors if present, and will also
run and test the examples included in the docstring.
Check the :ref:`spectrochempy docstring guide <docstring>` for a detailed guide
on how to format the docstring.

The examples in the docstring ('doctests') must be valid Python code,
that in a deterministic way returns the presented output, and that can be
copied and run by users. This can be checked with the script above, and is
also tested on github workflow testing. A failing doctest will be a blocker for merging a PR.
Check the :ref:`examples <docstring.examples>` section in the docstring guide
for some tips and tricks to get the doctests passing.

When doing a PR with a docstring update, it is good to post the
output of the validation script in a comment on github.


How to build the spectrochempy documentation
--------------------------------------------

Requirements
~~~~~~~~~~~~

First, you need to have a development environment to be able to build spectrochempy
(see the docs on :ref:`creating a development environment <contributing_environment>`).

Building the documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~
Navigate to your local ``docs/`` directory in the console and run::

    python make.py html

Then you can find the HTML output in the folder ``doc/build/html/`` .

The first time you build the docs, it will take quite a while because it has to run
all the code examples and build all the generated docstring pages. In subsequent
evocations, sphinx will try to only build the pages that have been modified
(The process remains however quite long!).

If you want to do a full clean build, do::

    python make.py clean
    python make.py html

Open the following file in a web browser to see the full documentation you
just built::

    docs/build/html/latest/index.html

And you'll see your new and improved documentation!

.. _contributing.dev_docs:

Building master branch documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When pull requests are merged into the spectrochempy ``master`` branch, the main parts of
the documentation are also built by a GitHub workflow. These docs are then hosted here: `www.spectrochempy.fr/latest/
<https://www.spectrochempy.fr/latest/>`__.

The stable version of the documentation, `www.spectrochempy.fr/stable/
<https://www.spectrochempy.fr/stable/>`__, is automatically build when a release created by the maintainers.
