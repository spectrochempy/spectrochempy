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

While commercial softwares generally provide powerful and easy-to-use Graphical User Interfaces (GUIs) for
data preprocessing and analysis, core algorithms such as baseline correction schemes, automatic subtractions,
peak detection, curve fitting etc.. are generally not disclosed to the user. Although they should be generally trusted
(it is the mere interest of these companies to provide reliable tools :-)), the devil is in the details and
sometimes, the spectroscopist needs knowing details on how her/his data are processed or analysed that are neither
given by the software manual nor disclosed by the company technical staff...

The "click, drag and drop" approach of many of GUIs also often prevents full reproducibility of data
processing and analysis. This is the case, for instance, of peak integration or baseline correction in many commercial
softwares. Moreover some of them do not always keep track of the data history, which is also be problematic for the
verification of data integrity and the reproducibility of results or scientific claims.

With this respect, spectrochempy is fully compliant with  pillars of open science: open source, open methodology, and
reproducibility. In particular, spectrochempy:

- is fully open source and relies on either public/published algorithms or algorithms developed internally but
  made open source and documented in the code.
- allows tracking the full history of datasets and analyses (`history` field of each datasets or object instances)
- allows embedding all content related to a particular work: raw data, scripts and notebooks for data processing
  and analysis, processed datasets, analyses, etc.. - in a dedicated data structure ('Project'), and hence warrants
  full reproducibility from raw data to final results.

Note that spectrochempy does not warrant 'data integrity' in the sense that the validity of raw data and the correct
use of processing and analysis tools belongs to the user responsability. Moreover, spectrochempy is not bug
free, and despite our efforts to benchmark it with well-known examples (e.g. from the literature, see the `Scpy Gallery
<https://www.spectrochempy.fr/gallery/auto_examples/index.html>`_), or internally with well established, commercial
softwares, we do not warrant that data processing is bug-free in some pecular (hopefuly rare) cases. In any event,
critical bugs affecting data integrity would be not only corrected, but also reported in the documentation of the scpy
release(s) affected, hence allowing users backtracking such errors.

A free software on a free platform
===================================
Even though powerful and reliable tools for spectroscopic data processing and analysis
have been developed by instrument manufacturers, software companies, or academic research groups, most of them are
proprietary and/or developed on proprietary computing environments (e.g. MATLAB). Although such proprietary softwares
are available in many laboratories through institutional licenses, this is not always the case. In some instances
the license cost can be problematic: developing countries, 'isolated' research groups, PhD or Post-docs leaving a
group with their data, etc...

It was thus important to us being able to share spectroscopic data / data processing tools / data analysis tools with
our colleagues, without imposing them a financial burden... or pushing them to use cracked copies...

WitIh this respect, spectrochempy is free of charge and under a free software license (`CeCILL version 2
<https://www.gnu.org/licenses/license-list.en.html#CeCILL>`_) retaining compatibility with
the GNU General Public License (GPL) and adapted to both international and French legal matters.

Powered by Python
=================
Among many programming languages for data science, the choice of Python has appeared natural as it is probably the most
popular, open-source language used by Data Scientists. It is extensible and can be run not only on Mac and Windows, but
also Unix/Linux, hence allowing using spectrochempy in a fully free software environment.

The Python eco-system of libraries and tools for data processing, analysis and vizualisation is huge
and fastly growing. With this respect, spectrochempy makes use of high-level, state-of-the-art libraries for numerical
calculations (numpy/scipy) and visualization (matplotlib, ..). As the Python community is extremely dynamic, we expect
that inexperienced (and highly experienced users as well :-), will easily find resources to solve particular needs
that would not be (yet) directly or completely fulfilled by spectrochempy.

We also count on interested members of this community (including you !), to contribute to the improvement of
spectrochempy through suggestions (<link>), bug reports (<link>) or code contributions (<link>)

Why NOT spectrochempy ?
=========================
You might NOT want to use spectrochempy if:

- you are fully satisfied with the software you are currently using. We do not challenge that :-)

- you are refractory to (Pyhton) code command-line or scripting. As spectrochempy is essentially an
  Application Programming Interface (API), it requires writing commands and small scripts. Still, Python
  (and hopefully spectrochempy) have a smooth learning curve. The basics of python, for instance, are teached in
  secondary schools. The spectrochempy documentation and resources contain fully documented examples and tutorials
  (see e.g. `Scpy Gallery <https://www.spectrochempy.fr/gallery/auto_examples/index.html>`_), which should be
  straightforward to transpose to your own data. In particular, spectrochempy provides
  jupyter notebooks, which mix texts, short blocks of codes and figures, so that the basic procedures (data import,
  basic processing and analyses, etc...) do not require much knowledge in programming.
  If however you remain refractory to spectrochempy (nobody's perfect, after all), we recommend you trying
  `Orange <https://orange.biolab.si/>`_, an excellent general purpose data mining and vizualisation pogram based on python,
  together with the `Orange-Spectroscopy Add-on <https://orange-spectroscopy.readthedocs.io/en/latest/>`_. It has less
  capability and features for advanced processing and analysis of spectriscopic data than spectrochempy but is much
  more intuitive for many other tasks.

- you work on very sensitive data (health, chemical safety, plant production, ...) and cannot afford the risk using a
  software still under developement and subject to bugs and probably frequent changes before 'maturity'. We agree and
  do not challenge that either :-)

- you work on spectroscopic data not easily treated which spectrochempy (currently mostly focused on optical
  spectroscopies and NMR) because some components or tools (e.g. import of your raw data, ...) are lacking: do not
  hesitate to contact us <link> or to suggest new features that should be added (<link>). We will consider all
  suggestions to make spectrochempy more broadly and more easily usable.

- "The heart has its reasons, of which reason knows nothing." We do not challenge that either, but open to hear your
  opinion and suggestions (<link>) !

