.. _whyscpy:

**********************
Why `SpectroChemPy`  ?
**********************

**Table of Contents**

.. contents::
   :local:

The `SpectroChemPy` project was developed to provide advanced spectroscopic data processing and analysis tools initially for internal purposes in the `LCS <https://www.lcs.ensicaen.fr/>`__, for the following reasons:

Designed for Open Science
=========================

While commercial software typically provides powerful and easy-to-use graphical user interfaces (GUIs), basic algorithms such as baseline correction schemes, automatic subtraction, peak detection, curve fitting, etc. are not typically disclosed to the user.

Although they can generally be trusted (it is in the sole interest of these companies to provide reliable tools), the spectroscopist sometimes needs to know the details of the processing or analysis of the data provided. These details are usually neither given in the software manual nor disclosed by the company's technical staff.

In addition, the "click, drag and drop" approach of many graphical user interfaces often prevents full reproducibility. This is the case, for example, with advanced integration or basic editing in many commercial software packages. Some of them do not always allow checking the data history, which is also problematic for checking data integrity and reproducibility.

In particular, `SpectroChemPy` :

- is fully open source and relies on algorithms documented in the code.
- allows following the complete history of data sets and analyses ("history" field)
- allows integrating the content of a particular job (data, scripts, notebooks, ...) into a dedicated data structure   ("Project"), guaranteeing a complete follow-up from the import of raw data to the final results..

.. note::

   SpectroChemPy does not guarantee "data integrity," in the sense that the validity of the raw data and the correct use of the processing and analysis tools are the responsibility of the user.

   SpectroChemPy is probably not free of bugs in some (hopefully rare) cases. In any case, critical bugs affecting data     integrity will be reported in the documentation for the version(s) of `SpectroChemPy` concerned, allowing the source of these errors to be traced.

Open Source Software on an Open-Source Platform
===============================================

Although powerful and popular tools have been developed by instrument manufacturers, software companies, or academic research groups, most are proprietary or require proprietary computing environments (e.g., MATLAB). The cost of licensing can be problematic in some cases: "isolated" research groups, developing countries, PhD or postdoc students leaving a group with their data, etc.

It was important for us to be able to share spectroscopic data / data processing tools / data analysis tools with our colleagues, without imposing them a financial burden... or pushing them to use cracked copies.

`SpectroChemPy` is free and under a free software license (`License [CeCILL-B] <https://cecill.info/index.en.html>`__). CeCILL-B follows the principle of the popular BSD license and its variants (Apache, X11 or W3C among others). In exchange for strong citation obligations (in all software incorporating a program covered by CeCILL-B and also through a Web site), the author authorizes the reuse of its software without any other constraints.

Powered by Python
=================

Python is probably the most popular open-source language used by Data Scientists. It is extensible and cross-platform, which allows `SpectroChemPy` to be used in an entirely free software environment. `SpectroChemPy` uses state-of-the-art libraries for numerical computation (`numpy <https://numpy.org/>`__, `scipy <https://www.scipy.org/>`__) and visualization (`matplotlib <https://matplotlib.org/>`__).

The Python community being extremely dynamic, all users should easily find resources to solve specific needs that are not (yet) directly or completely satisfied by `SpectroChemPy` .

We also count on motivated members of this community (of which you are a part), to contribute to the improvement of `SpectroChemPy` through (:ref:`contributing.bugs_report` ), or contributions to the code (:ref:`develguide` ).

Why NOT `SpectroChemPy` ?
=========================

You might **NOT** want to use `SpectroChemPy` if:

- You are averse to command line code (Python) or scripting. Since   `SpectroChemPy` is essentially an Application Programming Interface (API), it   requires writing commands and small scripts. Its documentation and resources contain fully documented   examples (see :ref:`examples-index`) and tutorials (see :ref:`userguide`),   which should be easy to translate into   your own data. In particular, the use of Jupyter notebooks mixing texts, code blocks and figures, so that basic   procedures (data import, basic processing and analysis, etc.) do not require much programming knowledge.

- you are working on spectroscopic data that are difficult to process with `SpectroChemPy` (currently mainly   focused on optical spectroscopy and NMR) because some components or tools (e.g., importing your raw data, ...) are   missing: please suggest new features that should be added (:ref:`contributing.bugs_report`). We will take into   consideration all suggestions to make `SpectroChemPy` more widely and easily usable.

- You are working on very sensitive data (health, chemical safety, plant production, ...) and cannot take the risk to   use software under development and subject to bugs and changes before   "maturity". We do not dispute that!

- You are fully satisfied with your current tools. "The heart has its reasons, of which the reason knows nothing." We   don't dispute that either, but we are open to your opinion and suggestions (:ref:`contributing.bugs_report`)!
