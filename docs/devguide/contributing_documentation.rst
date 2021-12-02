.. _contributing_documentation:

=================================
Contributing to the documentation
=================================

Contributing to the documentation is very important for the SpectroChemPy project.
We encourage you to help us improve the documentation, and you don't need to be a SpectroChemPy expert to do so!  If something in the doc doesn't make sense to you, update the the relevant section after you understand it is a great way to make sure it will help someone else.

.. contents:: Documentation:
   :local:


About the spectrochempy documentation
-------------------------------------

The documentation is written in **reStructuredText**, which is almost like writing
in plain English, and built using `Sphinx <https://www.sphinx-doc.org/en/master/>`__. The
Sphinx Documentation has an excellent `introduction to reST
<https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`__. Review the Sphinx docs to perform more
complex changes to the documentation as well.

Some other important things to know about the docs:

* The spectrochempy documentation consists of two parts: the docstrings in the code itself and the docs in folder ``docs/``.

  The docstrings provide a clear explanation of the usage of the individual
  functions, while the documentation in this folder consists of tutorial-like
  overviews per topic together with some other information (what's new,
  installation, etc).

* The docstrings follow the **Numpy Docstring
  Standard**. Follow the :ref:`spectrochempy docstring guide <docstring>` for detailed
  instructions on how to write a correct docstring.

  .. toctree::
     :maxdepth: 2

     contributing_docstring.rst

* The documentation, examples and tutorials make an heavy use of the `Jupyter Notebook <https://https://jupyter.org>`_.

  Almost all code examples in the docs are run (and the output saved) during the
  doc build. This approach means that code examples will always be up to date.

* Our API documentation files in ``docs/userguide/reference`` house the auto-generated
  documentation from the docstrings.

Todo : add some explanation the API reference generation

Updating a spectrochempy docstring
----------------------------------

When improving a single function or method's docstring, it is not necessarily
needed to build the full documentation (see next section).
Indeed, one can use a script that checks a docstring (for example for the ``spectrocopy.round`` method)::

    python scripts/validate_docstrings.py spectrochempy.round

This script will indicate some formatting errors if present, and will also
run and test the examples included in the docstring.
Check the :ref:`spectrochempy docstring guide <docstring>` for a detailed guide
on how to format the docstring.

The examples in the docstring ('doctests') must be valid Python code,
that in a deterministic way returns the presented output, and that can be
copied and run by users. This can be checked with the script above, and is
also tested on Travis. A failing doctest will be a blocker for merging a PR.
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

So how do you build the docs? Navigate to your local
``doc/`` directory in the console and run::

    python make.py html

Then you can find the HTML output in the folder ``doc/build/html/``.

The first time you build the docs, it will take quite a while because it has to run
all the code examples and build all the generated docstring pages. In subsequent
evocations, sphinx will try to only build the pages that have been modified.

If you want to do a full clean build, do::

    python make.py clean
    python make.py html

You can tell ``make.py`` to compile only a single section of the docs, greatly
reducing the turn-around time for checking your changes.

::

    # omit autosummary and API section
    python make.py clean
    python make.py --no-api

    # compile the docs with only a single section, relative to the "source" folder.
    # For example, compiling only this guide (doc/source/development/contributing.rst)
    python make.py clean
    python make.py --single development/contributing.rst

    # compile the reference docs for a single function
    python make.py clean
    python make.py --single spectrochempy.DataFrame.join

    # compile whatsnew and API section (to resolve links in the whatsnew)
    python make.py clean
    python make.py --whatsnew

For comparison, a full documentation build may take 15 minutes, but a single
section may take 15 seconds. Subsequent builds, which only process portions
you have changed, will be faster.

The build will automatically use the number of cores available on your machine
to speed up the documentation build. You can override this::

    python make.py html --num-jobs 4

Open the following file in a web browser to see the full documentation you
just built::

    doc/build/html/index.html

And you'll have the satisfaction of seeing your new and improved documentation!

.. _contributing.dev_docs:

Building master branch documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When pull requests are merged into the spectrochempy ``master`` branch, the main parts of
the documentation are also built by Travis-CI. These docs are then hosted `here
<https://spectrochempy.pydata.org/docs/dev/>`__, see also
the :any:`Continuous Integration <contributing.ci>` section.

Previewing changes
------------------

Once, the pull request is submitted, GitHub Actions will automatically build the
documentation. To view the built site:

#. Wait for the ``CI / Web and docs`` check to complete.
#. Click ``Details`` next to it.
#. From the ``Artifacts`` drop-down, click ``docs`` or ``website`` to download
   the site as a ZIP file.
