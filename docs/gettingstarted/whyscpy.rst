.. _whyscpy:

Why spectrochempy  ?
====================

**Table of contents**

.. contents::
   :local:

The |scpy| project has been developed to provide advanced tools for the processing and analysis of
spectroscopic data, initially for internal needs in the `LCS <https://www.lcs.ensicaen.fr/>`_, for the following
reasons:

Designed for Open Science
-------------------------

While commercial software usually provides powerful and easy-to-use graphical user interfaces (GUIs), basic algorithms
such as baseline correction systems, automatic subtraction, peak detection, curve fitting, etc. are generally not
disclosed to the user.

Although they should generally be trusted (it is in the sole interest of these companies to provide reliable tools),
sometimes the spectroscopist needs to know the details of how his data are processed or analyzed.
These details are usually neither given in the software manual nor disclosed by the company's technical staff .

In addition, the "click, drag and drop" approach of many graphical interfaces often prevents full reproducibility.
This is the case, for example, with advanced integration or basic editing in many commercial software packages.
Some of them do not always hold monitoring of data history, which is also problematic for the verification of data
integrity and reproducibility.

The use of the |scpy| is consistent with the three pillars of open science:
open source, open methodology and replicability.

In particular, |scpy| :

- is entirely open source and relies on algorithms documented in the code.
- allows the complete history of data sets and analyses to be tracked ("history" field)
- allows to integrate the content of a particular job (data, scripts, notebooks, ...) into a dedicated data structure
  ("Project"), guaranteeing a complete follow-up from the import of raw data to the final results.

.. Note::

    |scpy| does not warrant 'data integrity' in the sense that the validity of raw data and the correct
    use of processing and analysis tools belongs to the user responsibility.

    |scpy| is probably not bug-free in some peculiar (hopefully rare) cases. In any event, critical bugs
    affecting data integrity will be reported in the documentation of the |scpy| release(s) affected, hence allowing
    backtracking such errors.

A free software on a free platform
----------------------------------

If powerful and popular tools have been developed by instrument manufacturers, software companies, or academic research
groups, most of them are proprietary or require proprietary computing environments (e.g., MATLAB).
The license cost can be problematic in some cases: 'isolated' research groups, developing countries, PhD or
Post-docs leaving a group with their data, etc...

It was important to us being able to share spectroscopic data / data processing tools / data analysis tools with
our colleagues, without imposing them a financial burden... or pushing them to use cracked copies.

|scpy| is free of charge and under a free software license (:ref:`license` ) retaining compatibility with
the GNU General Public License (GPL) and adapted to both international and French legal matters.

Powered by Python
-----------------

Python is probably the most popular, open-source language used by Data Scientists. It is extensible and is
cross-platform hence allowing using spectrochempy in a fully free software environment. |scpy| makes use of
state-of-the-art libraries for numerical calculations (`numpy <https://numpy.org/>`_ , `scipy <https://www.scipy.org/>`_)
and visualization (`matplotlib <https://matplotlib.org/>`_).

As the Python community is extremely dynamic, all users should easily find resources to solve particular needs
that would not be (yet) directly or completely fulfilled by spectrochempy.

We also count on motivated members of this community (including you), to contribute to the improvement of
spectrochempy through  bug reports and enhancement requests (:ref:`contributing.bug_reports`),
or contributions to the code (:ref:`develguide`)

Why NOT spectrochempy ?
-----------------------

You might **NOT** want to use spectrochempy if:

- you are refractory to (Python) code command line or scripting. As spectrochempy is essentially an
  Application Programming Interface (API), it requires writing commands and small scripts. Still, Python
  and spectrochempy have a smooth learning curve.The spectrochempy documentation and resources contain fully documented
  examples and tutorials (see :ref:`tutorials` and :ref:`sphx_glr_gallery_auto_examples`), which should be   straightforward to transpose to your
  own data. In particular, Jupyter notebooks, mixing texts, blocks of codes and figures, so that the basic procedures
  (data import, basic processing and analyses, etc...) do not require much knowledge in programming.

- you work on spectroscopic data not easily treated which spectrochempy (currently mostly focused on optical
  spectroscopies and NMR) because some components or tools (e.g., import of your raw data, ...) are lacking: do not
  to suggest new features that should be added (:ref:`contributing.bug_reports`). We will consider all
  suggestions to make spectrochempy more broadly and more easily usable.

- you work on very sensitive data (health, chemical safety, plant production, ...) and cannot afford the risk using software under development and subject to bugs and changes before 'maturity'. We do not challenge that !

- you are fully satisfied by your current tools. "The heart has its reasons, of which reason knows nothing." We do not
  challenge that either, but open to hear your opinion and suggestions (:ref:`contributing.bug_reports`) !


