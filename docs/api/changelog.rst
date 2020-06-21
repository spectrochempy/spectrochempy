What's new
===========

.. START CHANGELOG





Version 0.1.20
-----------------------------------

Bugs fixed
~~~~~~~~~~~

* FIX `#116 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/116>`_ : CI: try to fix bug on deploy
* FIX `#115 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/115>`_ : CI: revise the changelog script in make.py
* FIX `#87 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/87>`_ : Check for update not working
* FIX `#76 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/76>`_ : read_opus() shifts the xaxis
* FIX `#27 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/27>`_ : Solve pint version > 0.9 incompatibilities
* FIX `#17 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/17>`_ : Doc building using sphinx generates some warnings

Features added
~~~~~~~~~~~~~~~~

* `#114 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/114>`_ : CI: Travis deploy and build doc improved 
* `#111 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/111>`_ : TravisCI automatic uploading to pypi 
* `#110 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/110>`_ : TravisCI to build docs automatically
* `#105 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/105>`_ : CI: continuing setup
* `#102 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/102>`_ : Travis docs building
* `#101 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/101>`_ : Feature/travis deploy - taking into account restrictions regarding secure keys
* `#98 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/98>`_ : PEP8 + Travis CI setup
* `#93 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/93>`_ : find_peaks() should makes use of Coord instead on indices 
* `#90 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/90>`_ : make smooth() and savgol_filter() more consistent
* `#46 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/46>`_ : Tutorials on IR data import

Tasks terminated
~~~~~~~~~~~~~~~~~

* `#75 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/75>`_ : Automate building of docs for new release and dev version.
* `#49 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/49>`_ : Remove nmrglue from spectrochempy/extern



Version 0.1.19
---------------------

Bugs fixed
~~~~~~~~~~~

* FIX `#74 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/74>`_ : Fix warning issued during tests
* FIX `#69 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/69>`_ : allow passing description to read_omnic
* FIX `#65 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/65>`_ : Installation of matplotlib styles
* FIX `#63 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/63>`_ : The file selector does not work when the path has a final slash.
* FIX `#59 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/59>`_ : Conda install is not working on linux (sometimes!)
* FIX `#48 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/48>`_ : Deprecation warning with numpy elementwise == comparison 

Features added
~~~~~~~~~~~~~~~~

* `#67 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/67>`_ : MCR-ALS improved 
* `#66 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/66>`_ : improve changelog display

Tasks terminated
~~~~~~~~~~~~~~~~~

* `#70 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/70>`_ : Set rules on Github for new commits
* `#50 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/50>`_ : Set up of workflows when pushing on GitHub
* `#40 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/40>`_ : Automate building and deployment of new releases



Version 0.1.18
---------------------

Bugs fixed
~~~~~~~~~~~

* FIX `#64 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/64>`_ : Install of the matplotlib styles doesn't work during setup
* FIX `#62 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/62>`_ : Generic read function do not open a dialog when called without argument
* FIX `#58 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/58>`_ : Wrong instruction in install guides (MAC and WIN

Features added
~~~~~~~~~~~~~~~~

* `#54 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/54>`_ : implement the numpy equivalent method `astype`
* `#53 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/53>`_ : Add new numpy equivalent functions such as np.eyes, np.identity.
* `#45 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/45>`_ : Bruker Opus Import

Tasks terminated
~~~~~~~~~~~~~~~~~

* `#61 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/61>`_ : Make a repository on github. Get a DOI number for it. 
* `#56 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/56>`_ : Add nmrglue to our conda channel - and remove it from extern
* `#52 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/52>`_ : Add  brukeropusreader to our conda channel
* `#51 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/51>`_ : Add a linux quadprog version into our conda channel



Version 0.1.17
---------------------

Bugs fixed
~~~~~~~~~~~

* FIX `#44 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/44>`_ : TQDM generate errors during doc building in examples.
* FIX `#38 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/38>`_ : Tutorial notebooks that contain a dialog for filename do not run silently during sphinx build.
* FIX `#37 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/37>`_ : QT error in doc
* FIX `#33 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/33>`_ : Size of the figures in pdf documentation often too wide. 
* FIX `#30 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/30>`_ : Fix doctrings and rst files  so that the pdf manual get correct with titles and sections
* FIX `#28 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/28>`_ : loose coord  when slicing by integer array
* FIX `#26 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/26>`_ : Test Console don't pass on WINDOWS
* FIX `#23 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/23>`_ : pca reconstruction for an omnic dataset
* FIX `#15 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/15>`_ : Fix doc RST syntax

Features added
~~~~~~~~~~~~~~~~

* `#42 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/42>`_ : Add a progress bar during loading of the library 
* `#39 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/39>`_ : make changelog automatic when making the doc
* `#35 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/35>`_ : Check for new version at the program start up
* `#32 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/32>`_ : The autosub function does not return the subtraction coefficients
* `#16 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/16>`_ : Create an importer to get the issues from Bitbucket and start the issue tracker here.

Tasks terminated
~~~~~~~~~~~~~~~~~

* `#29 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/29>`_ : import data: tutorial, examples, tests
* `#25 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/25>`_ : Conda Recipe
* `#13 <https://api.github.com/repos/spectrochempy/spectrochempy/issues/13>`_ : Redmine website configuration



Version 0.1.16
---------------

*  Initial version released as pypi and conda package



Versions 0.1.0
---------------

* initial Development version


