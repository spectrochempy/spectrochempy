What\'s new
===========

Version 0.1.20
--------------

### Bugs fixed

-   FIX
    [\#116](https://api.github.com/repos/spectrochempy/spectrochempy/issues/116)
    : CI: try to fix bug on deploy
-   FIX
    [\#115](https://api.github.com/repos/spectrochempy/spectrochempy/issues/115)
    : CI: revise the changelog script in make.py
-   FIX
    [\#87](https://api.github.com/repos/spectrochempy/spectrochempy/issues/87)
    : Check for update not working
-   FIX
    [\#76](https://api.github.com/repos/spectrochempy/spectrochempy/issues/76)
    : read\_opus() shifts the xaxis
-   FIX
    [\#27](https://api.github.com/repos/spectrochempy/spectrochempy/issues/27)
    : Solve pint version \> 0.9 incompatibilities
-   FIX
    [\#17](https://api.github.com/repos/spectrochempy/spectrochempy/issues/17)
    : Doc building using sphinx generates some warnings

### Features added

-   [\#114](https://api.github.com/repos/spectrochempy/spectrochempy/issues/114)
    : CI: Travis deploy and build doc improved
-   [\#111](https://api.github.com/repos/spectrochempy/spectrochempy/issues/111)
    : TravisCI automatic uploading to pypi
-   [\#110](https://api.github.com/repos/spectrochempy/spectrochempy/issues/110)
    : TravisCI to build docs automatically
-   [\#105](https://api.github.com/repos/spectrochempy/spectrochempy/issues/105)
    : CI: continuing setup
-   [\#102](https://api.github.com/repos/spectrochempy/spectrochempy/issues/102)
    : Travis docs building
-   [\#101](https://api.github.com/repos/spectrochempy/spectrochempy/issues/101)
    : Feature/travis deploy - taking into account restrictions regarding
    secure keys
-   [\#98](https://api.github.com/repos/spectrochempy/spectrochempy/issues/98)
    : PEP8 + Travis CI setup
-   [\#93](https://api.github.com/repos/spectrochempy/spectrochempy/issues/93)
    : find\_peaks() should makes use of Coord instead on indices
-   [\#90](https://api.github.com/repos/spectrochempy/spectrochempy/issues/90)
    : make smooth() and savgol\_filter() more consistent
-   [\#46](https://api.github.com/repos/spectrochempy/spectrochempy/issues/46)
    : Tutorials on IR data import

### Tasks terminated

-   [\#75](https://api.github.com/repos/spectrochempy/spectrochempy/issues/75)
    : Automate building of docs for new release and dev version.
-   [\#49](https://api.github.com/repos/spectrochempy/spectrochempy/issues/49)
    : Remove nmrglue from spectrochempy/extern

Version 0.1.19
--------------

### Bugs fixed

-   FIX
    [\#74](https://api.github.com/repos/spectrochempy/spectrochempy/issues/74)
    : Fix warning issued during tests
-   FIX
    [\#69](https://api.github.com/repos/spectrochempy/spectrochempy/issues/69)
    : allow passing description to read\_omnic
-   FIX
    [\#65](https://api.github.com/repos/spectrochempy/spectrochempy/issues/65)
    : Installation of matplotlib styles
-   FIX
    [\#63](https://api.github.com/repos/spectrochempy/spectrochempy/issues/63)
    : The file selector does not work when the path has a final slash.
-   FIX
    [\#59](https://api.github.com/repos/spectrochempy/spectrochempy/issues/59)
    : Conda install is not working on linux (sometimes!)
-   FIX
    [\#48](https://api.github.com/repos/spectrochempy/spectrochempy/issues/48)
    : Deprecation warning with numpy elementwise == comparison

### Features added

-   [\#67](https://api.github.com/repos/spectrochempy/spectrochempy/issues/67)
    : MCR-ALS improved
-   [\#66](https://api.github.com/repos/spectrochempy/spectrochempy/issues/66)
    : improve changelog display

### Tasks terminated

-   [\#70](https://api.github.com/repos/spectrochempy/spectrochempy/issues/70)
    : Set rules on Github for new commits
-   [\#50](https://api.github.com/repos/spectrochempy/spectrochempy/issues/50)
    : Set up of workflows when pushing on GitHub
-   [\#40](https://api.github.com/repos/spectrochempy/spectrochempy/issues/40)
    : Automate building and deployment of new releases

Version 0.1.18
--------------

### Bugs fixed

-   FIX
    [\#64](https://api.github.com/repos/spectrochempy/spectrochempy/issues/64)
    : Install of the matplotlib styles doesn\'t work during setup
-   FIX
    [\#62](https://api.github.com/repos/spectrochempy/spectrochempy/issues/62)
    : Generic read function do not open a dialog when called without
    argument
-   FIX
    [\#58](https://api.github.com/repos/spectrochempy/spectrochempy/issues/58)
    : Wrong instruction in install guides (MAC and WIN

### Features added

-   [\#54](https://api.github.com/repos/spectrochempy/spectrochempy/issues/54)
    : implement the numpy equivalent method [astype]{.title-ref}
-   [\#53](https://api.github.com/repos/spectrochempy/spectrochempy/issues/53)
    : Add new numpy equivalent functions such as np.eyes, np.identity.
-   [\#45](https://api.github.com/repos/spectrochempy/spectrochempy/issues/45)
    : Bruker Opus Import

### Tasks terminated

-   [\#61](https://api.github.com/repos/spectrochempy/spectrochempy/issues/61)
    : Make a repository on github. Get a DOI number for it.
-   [\#56](https://api.github.com/repos/spectrochempy/spectrochempy/issues/56)
    : Add nmrglue to our conda channel - and remove it from extern
-   [\#52](https://api.github.com/repos/spectrochempy/spectrochempy/issues/52)
    : Add brukeropusreader to our conda channel
-   [\#51](https://api.github.com/repos/spectrochempy/spectrochempy/issues/51)
    : Add a linux quadprog version into our conda channel

Version 0.1.17
--------------

### Bugs fixed

-   FIX
    [\#44](https://api.github.com/repos/spectrochempy/spectrochempy/issues/44)
    : TQDM generate errors during doc building in examples.
-   FIX
    [\#38](https://api.github.com/repos/spectrochempy/spectrochempy/issues/38)
    : Tutorial notebooks that contain a dialog for filename do not run
    silently during sphinx build.
-   FIX
    [\#37](https://api.github.com/repos/spectrochempy/spectrochempy/issues/37)
    : QT error in doc
-   FIX
    [\#33](https://api.github.com/repos/spectrochempy/spectrochempy/issues/33)
    : Size of the figures in pdf documentation often too wide.
-   FIX
    [\#30](https://api.github.com/repos/spectrochempy/spectrochempy/issues/30)
    : Fix doctrings and rst files so that the pdf manual get correct
    with titles and sections
-   FIX
    [\#28](https://api.github.com/repos/spectrochempy/spectrochempy/issues/28)
    : loose coord when slicing by integer array
-   FIX
    [\#26](https://api.github.com/repos/spectrochempy/spectrochempy/issues/26)
    : Test Console don\'t pass on WINDOWS
-   FIX
    [\#23](https://api.github.com/repos/spectrochempy/spectrochempy/issues/23)
    : pca reconstruction for an omnic dataset
-   FIX
    [\#15](https://api.github.com/repos/spectrochempy/spectrochempy/issues/15)
    : Fix doc RST syntax

### Features added

-   [\#42](https://api.github.com/repos/spectrochempy/spectrochempy/issues/42)
    : Add a progress bar during loading of the library
-   [\#39](https://api.github.com/repos/spectrochempy/spectrochempy/issues/39)
    : make changelog automatic when making the doc
-   [\#35](https://api.github.com/repos/spectrochempy/spectrochempy/issues/35)
    : Check for new version at the program start up
-   [\#32](https://api.github.com/repos/spectrochempy/spectrochempy/issues/32)
    : The autosub function does not return the subtraction coefficients
-   [\#16](https://api.github.com/repos/spectrochempy/spectrochempy/issues/16)
    : Create an importer to get the issues from Bitbucket and start the
    issue tracker here.

### Tasks terminated

-   [\#29](https://api.github.com/repos/spectrochempy/spectrochempy/issues/29)
    : import data: tutorial, examples, tests
-   [\#25](https://api.github.com/repos/spectrochempy/spectrochempy/issues/25)
    : Conda Recipe
-   [\#13](https://api.github.com/repos/spectrochempy/spectrochempy/issues/13)
    : Redmine website configuration

Version 0.1.16
--------------

-   Initial version released as pypi and conda package

Versions 0.1.0
--------------

-   initial Development version
