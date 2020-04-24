.. _whyscpy:

Why spectrochempy  ?
######################

**Table of contents**

.. contents::
   :local:

The spectrochempy project (|scpy|) has been developed to provide advanced tools for the processing and analysis of
spectroscopic data, initially for internal needs in the `LCS <https://www.lcs.ensicaen.fr/>`_, for the the following
reasons:

Designed for Open Science
=========================

While commercial softwares generally provide powerful and easy-to-use Graphical User Interfaces (GUIs), core algorithms
such as baseline correction schemes, automatic subtractions, peak detection, curve fitting etc.. are generally not
disclosed to the user. Although they should be generally trusted (it is the mere interest of these companies to provide
reliable tools), sometimes, the spectroscopist needs knowing details on how her/his data are processed or analysed that
are neither given by the software manual nor disclosed by the company technical staff...

The "click, drag and drop" approach of many of GUIs also often prevents full reproducibility. This is the case, for
instance, of peak integration or baseline correction in many commercial softwares. Some of them do not always keep
track of the data history, which is also be problematic for the verification of data integrity and the reproducibility.

Spectrochempy is compliant with 3 pillars of open science: open source, open methodology, and reproducibility.
In particular, spectrochempy:

- is fully open source and relies on algorithms documented in the code.
- allows tracking the full history of datasets and analyses (`history` fields)
- allows embedding content of particular work (data, scripts, notebooks, ...)in a dedicated data structure ('Project'),
  warranting full tracking from raw data import to final results.

.. Note::
    Spectrochempy does not warrant 'data integrity' in the sense that the validity of raw data and the correct
    use of processing and analysis tools belongs to the user responsability.

    Spectrochempy is not probably not bug-free in some pecular (hopefuly rare) cases. In any event, critical bugs
    affecting data integrity will be reported in the documentation of the scpy release(s) affected, hence allowing
    backtracking such errors.

A free software on a free platform
===================================
If powerful and popular tools have been developed by instrument manufacturers, software companies, or academic research
groups, most of them are proprietary or require proprietary computing environments (e.g. MATLAB).
The license cost can be problematic in dome instances: 'isolated' research groups, developing countries, PhD or
Post-docs leaving a group with their data, etc...

It was important to us being able to share spectroscopic data / data processing tools / data analysis tools with
our colleagues, without imposing them a financial burden... or pushing them to use cracked copies.

Spectrochempy is free of charge and under a free software license (`CeCILL version 2
<https://www.gnu.org/licenses/license-list.en.html#CeCILL>`_) retaining compatibility with
the GNU General Public License (GPL) and adapted to both international and French legal matters.

Powered by Python
=================
Python is probably the most popular, open-source language used by Data Scientists. It is extensible and is
cross-platform hence allowing using spectrochempy in a fully free software environment. Sspectrochempy makes use of
state-of-the-art libraries for numerical calculations (`numpy <https://numpy.org/>`_ , `scipy <https://www.scipy.org/>`_)
and visualization (`matplotlib <https://matplotlib.org/>`_).

As the Python community is extremely dynamic, all users should easily find resources to solve particular needs
that would not be (yet) directly or completely fulfilled by spectrochempy.

We also count on motivated members of this community (including you), to contribute to the improvement of
spectrochempy through  bug reports and enhancement requests (:ref:`contributing.bug_reports`),
or contributions to the code (:ref:`develguide`)

Why NOT spectrochempy ?
=========================
You might NOT want to use spectrochempy if:

- you are refractory to (Pyhton) code command-line or scripting. As spectrochempy is essentially an
  Application Programming Interface (API), it requires writing commands and small scripts. Still, Python
  and spectrochempy have a smooth learning curve.The spectrochempy documentation and resources contain fully documented
  examples and tutorials (see :ref:`tutorials` and :ref:`sphx_glr_gallery_auto_examples`), which should be   straightforward to transpose to your
  own data. In particular, Jupyter notebooks, mixing texts, blocks of codes and figures, so that the basic procedures
  (data import, basic processing and analyses, etc...) do not require much knowledge in programming.

- you work on spectroscopic data not easily treated which spectrochempy (currently mostly focused on optical
  spectroscopies and NMR) because some components or tools (e.g. import of your raw data, ...) are lacking: do not
  to suggest new features that should be added (:ref:`contributing.bug_reports`). We will consider all
  suggestions to make spectrochempy more broadly and more easily usable.

- you work on very sensitive data (health, chemical safety, plant production, ...) and cannot afford the risk using a
  software under developement and subject to bugs and changes before 'maturity'. We do not challenge that !

- you are fully satisfied by your current tools. "The heart has its reasons, of which reason knows nothing." We do not
  challenge that either, but open to hear your opinion and suggestions (:ref:`contributing.bug_reports`) !

