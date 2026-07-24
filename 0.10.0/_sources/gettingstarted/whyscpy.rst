.. _whyscpy:

**********************
Why SpectroChemPy?
**********************

**Table of Contents**

.. contents:: Contents
   :local:
   :depth: 2

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

Open Source and Free
====================

`SpectroChemPy` is free software under the `CeCILL-B License <https://cecill.info/index.en.html>`__,
similar to BSD licenses. The license requires:

* Strong citation requirements
* Attribution in derivative works
* Web presence attribution

In return, users can freely use and modify the software.

Python-Powered Analysis
=======================

Benefits of using Python:

* Popular in data science
* Cross-platform compatibility
* Extensive scientific libraries (numpy, scipy, matplotlib)
* Active community support
* Easy integration with other tools

Why NOT SpectroChemPy ?
=======================

You might **NOT** want to use `SpectroChemPy` if:

- You are averse to command line code (Python) or scripting. Since   `SpectroChemPy` is essentially an Application Programming Interface (API), it   requires writing commands and small scripts. Its documentation and resources contain fully documented   examples (see :ref:`examples-index`) and tutorials (see :ref:`userguide`),   which should be easy to translate into   your own data. In particular, the use of Jupyter notebooks mixing texts, code blocks and figures, so that basic   procedures (data import, basic processing and analysis, etc.) do not require much programming knowledge.

- you are working on spectroscopic data that are difficult to process with `SpectroChemPy` (currently mainly   focused on optical spectroscopy and NMR) because some components or tools (e.g., importing your raw data, ...) are   missing: please suggest new features that should be added (:ref:`contributing.bugs_report`). We will take into   consideration all suggestions to make `SpectroChemPy` more widely and easily usable.

- You are working on very sensitive data (health, chemical safety, plant production, ...) and cannot take the risk to   use software under development and subject to bugs and changes before   "maturity". We do not dispute that!

- You are fully satisfied with your current tools. "The heart has its reasons, of which the reason knows nothing." We   don't dispute that either, but we are open to your opinion and suggestions (:ref:`contributing.bugs_report`)!
