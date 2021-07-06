.. _whyscpy:

****************************************
Why |scpy|  ?
****************************************

**Table of contents**

.. contents::
   :local:

The |scpy| project was developed to provide advanced tools for processing and
analyzing spectroscopic data, initially for internal purposes in the
`LCS <https://www.lcs.ensicaen.fr/>`__, for the following reasons:

Designed for Open Science
==========================

While commercial software usually provides powerful and easy-to-use graphical user interfaces (GUIs), basic algorithms
such as baseline correction systems, automatic subtraction, peak detection, curve fitting, etc. are generally not
disclosed to the user.

Although they are generally to be trusted (it is in the sole interest of these companies to provide reliable tools),
the spectroscopist sometimes needs to know the details of how the data provided is processed or analyzed. These details
are usually neither given in the software manual nor disclosed by the company's technical staff.

In addition, the "click, drag and drop" approach of many graphical interfaces often prevents full reproducibility. This
is the case, for example, with advanced integration or basic editing in many commercial software products. Some of them
do not always allow to check the history of the data, which is also problematic for the verification of data integrity
and reproducibility.

In particular, |scpy| :

- is entirely open source and relies on algorithms documented in the code.
- allows to follow the complete history of data sets and analyses ("history" field)
- allows to integrate the content of a particular job (data, scripts, notebooks, ...) into a dedicated data structure
  ("Project"), guaranteeing a complete follow-up from the import of raw data to the final results..

.. note::

    SpectroChemPy does not guarantee "data integrity", in that the validity of the raw data and the correct use of the
    processing and analysis tools is the responsibility of the user.

    SpectroChemPy is probably not free of bugs in some (hopefully rare) cases. In any case, critical bugs affecting data
    integrity will be reported in the documentation for the version(s) of |scpy| concerned, allowing the source
    of these errors to be traced.

Open souce software on an open source platform
===============================================

While powerful and popular tools have been developed by instrument manufacturers, software companies, or academic
research groups, most are proprietary or require proprietary computing environments (e.g., MATLAB). The cost of
licensing can be problematic in some cases: "isolated" research groups, developing countries, doctoral or post-doc
students leaving a group with their data, etc.

It was important for us to be able to share spectroscopic data / data processing tools / data analysis tools with our
colleagues, without imposing a financial burden on them... or pushing them to use cracked copies.

|scpy| is free of charge and under a free software license
(`Licence [CeCILL-B] <https://cecill.info/index.en.html>`__). CeCILL-B follows the principle of the popular BSD license
and its variants (Apache, X11 or W3C among others). In exchange for strong citation obligations (in all software
incorporating a program covered by CeCILL-B and also through a Web site), the author authorizes the reuse of its
software without any other constraints.

Powered by Python
==================

Python is probably the most popular, open-source language used by Data Scientists. It is extensible and cross-platform,
allowing |scpy| to be used in an entirely open-source software environment. |scpy| uses state-of-the-art
libraries for numerical computation (`numpy <https://numpy.org/>`__, `scipy <https://www.scipy.org/>`__) and
visualization (`matplotlib <https://matplotlib.org/>`__).

As the Python community is extremely dynamic, all users should easily find resources to solve specific needs that are
not (yet) directly or completely satisfied by |scpy|.

We also rely on the motivated members of this community (of which you are a part), to contribute to the improvement of
|scpy| through (:ref:`contributing.bugs_report`), or contributions to the code (:ref:`develguide`).

Why NOT |scpy| ?
========================

You might **NOT** want to use |scpy| if:

- you are resistant to command line code (Python) or scripts. Since
  |scpy| is essentially an Application Programming Interface (API), it
  requires writing commands and small scripts. its documentation and resources contain fully documented
  examples (see :ref:`examples-index`) and tutorials (see :ref:`userguide`),  which should be easy to transpose into
  your own data. In particular, the use of Jupyter notebooks mixing texts, code blocks and figures, so that basic
  procedures (data import, basic processing and analysis, etc...) do not require much programming knowledge.

- you are working on spectroscopic data that are difficult to process with |scpy| (currently mainly
  focused on optical spectroscopy and NMR) because some components or tools (e.g. importing your raw data, ...) are
  missing: please suggest new features that should be added (:ref:`contributing.bugs_report`). We will take into
  consideration all suggestions to make |scpy| more widely and easily usable.

- you are working on very sensitive data (health, chemical safety, plant production, ...) and cannot take the risk to
  use a software under development and subject to bugs and changes before "maturity". We do not dispute this!

- you are fully satisfied with your current tools. "The heart has its reasons, of which the reason knows nothing". We
  don't dispute that either, but we are open to your opinion and suggestions (:ref:`contributing.bugs_report`)!
