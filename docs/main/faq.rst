.. _faq:

..
    todo we could just make a link to redmine FAQ forum?

Frequently Asked Questions (FAQ)
=================================

.. _faq_preference_file:

Where are the Preference Files Saved?
--------------------------------------
Typically, the main application preference file is saved in a hidden directory
located in your home user directory: ``$HOME/.spectrochempy/config``. but if the
``SCP_CONFIG_HOME`` environment variable is set and the
``$SCP_CONFIG_HOME/spectrochempy`` directory exists, it will be that
directory.

In principle you should not need to access files in this directory, but if you wants to do it,
you can use one of these solutions :

On Mac OSX system you access to this file by typing in the terminal:

.. sourcecode:: bash

    $ cd ~/.spectrochempy
    $ open spectrochempy.ini

On Linux system, the second command can be replaced by:

.. sourcecode:: bash

    $ vim spectrochempy.ini

or whatever you prefer to read and edit a text file.

.. _terminal_not_sourcing_profile:

in PyCharm on OSX, the terminal is not sourcing .bash_profile
--------------------------------------------------------------

Change the following in Pycherm/preferences/terminal:

`/bin/bash`  by  `/bin/bash --rcfile ~/.bash_profile`

.. image:: images/bashprofile.png
  :width: 500 px
  :alt: bash_profile setting for terminal
  :align: center


in Pycharm on OSX, latex is not found from the terminal
--------------------------------------------------------

While it is found on a shell terminal, it doesn't work in pycharm.

For exemple when executing a notebook from PyCharm, one  can get:

`FileNotFoundError: [Errno 2] No such file or directory: kpsewhich ...`


**Solution**: add the path to the latex command in the `.bash_profile`:

'/Library/TeX/texbin/'

