.. _faq:

Frequently Asked Questions (FAQ)
#################################

.. _faq_preference_file:

Where are the Preference Files Saved?
=====================================
Typically, the main application preference file is saved in a hidden directory
located in your home user directory: ``$HOME/.spectrochempy/config``. but if the
``SCP_CONFIG_HOME`` environment variable is set and the
``$SCP_CONFIG_HOME/spectrochempy`` directory exists, it will be that
directory.
In principle you
should not need to access files in this directory, but if you wants to do it,
you can use one of these solutions :

On Mac OSX system you access to this file by typing in the terminal:

.. sourcecode:: bash

	$ cd ~/.spectrochempy
	$ open spectrochempy.ini

On Linux system, the second command can be replaced by:

.. sourcecode:: bash

	$ vim spectrochempy.ini

or whatever you prefer to read and edit a text file.

.. _faq_cannot_launch_jupyter_from_terminal:


