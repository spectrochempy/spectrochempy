# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.9.1
#   widgets:
#     application/vnd.jupyter.widget-state+json:
#       state: {}
#       version_major: 2
#       version_minor: 0
# ---

# %% [markdown]
# ## API Configuration
#
# Many options of the API can be set up.
# Let's first import it in the u the usual way:

# %%
import spectrochempy as scp

# %% [markdown]
# ### General information
# General informations on the API can be obtained the following variables:

# %%
print(f'   copyright : {scp.copyright}')
print(f'     version : {scp.version}')
print(f'     release : {scp.release}')
print(f'     license : {scp.license}')
print(f'         url : {scp.url}')
print(f'release_date : {scp.release_date}')
print(f'     authors : {scp.authors}')
print(f'contributors : {scp.contributors}')
print(f' description : {scp.description}')

# %% [markdown]
# ### Loglevel
# During the execution, the API can display, besides the expected output, various messages categorized according
# to their criticality:
#
# | Loglevel / criticality | use |
# | --- | --- |
# | `DEBUG` / `10` | help diagnose problems on the running process or help developers |
# | `INFO`  / `20` | general information on the running process |
# | `WARNING` / `30` (default) | a condition might cause a problem with respect to the expected behaviour |
# | `ERROR`   / `40` | wrong argument/commands are used or bug |
# | `CRITICAL` / `50` | the process could lead to a system crash |
#
#  Not all these informations are always necessary and the level of information displayed by SpectroChemPy can be
#  tuned using the command `scp.set_loglevel()` with the rule that only informations having a
#  criticity larger than that passed to the `set_loglevel()` function will be shown.
#
#  For instance, the `DEBUG` level can be triggered by using one of the three equivalent instructions
# ```python
# scp.set_loglevel('DEBUG')
# scp.set_loglevel(scp.DEBUG)
# scp.set_loglevel(10)
# ```
# The current loglevel can be obtained with the `scp.get_loglevel()` function.
#
# The following instruction prints the current loglevel
# %%
print(f'Default: {scp.get_loglevel()}')       # print the current loglevel

# %% [markdown]
# As expected, it yields `30` the numerical value corresponding to the `WARNING` level. Now, the next instruction
# lowers the loglevel to `ÌNFO`:
# %%
scp.set_loglevel(scp.INFO)                    # set loglevel to 'INFO'


# %% [markdown]
# We see that the API then delivers the INFO message: `changed default log_level to INFO`.
#
# The next instruction prints the current level. As expected, it yields `20` the numerical value corresponding to
# the `INFO` level:.
# %%
print(f'New loglevel: {scp.get_loglevel()}')

# %% [markdown]
# And  finally, the next instructions reset the loglevel to `WARNING`, and print it.  and return to the default level.
# As seen below, no message `changed default log_level to ...` is delivered

# %%
scp.set_loglevel('WARNING')                   #reset to default
print(f'New loglevel: {scp.get_loglevel()}')

# %% [markdown]
# It is also possible to issue such messages in scripts. In the cell below, we set the loglevel to `INFO` and try to
# print two types of messages:

# %%
scp.set_loglevel('INFO')
scp.info_('this is an info message!')
scp.debug_('this is a debug message!')

# %% [markdown]
# A expected, only the info message was displayed.
#
# If we change the loglevel to ``DEBUG``, then the two messages will be printed:

# %%
scp.set_loglevel(scp.DEBUG)
scp.info_('this is an info message!')
scp.debug_('this is a debug message!')

# %% [markdown]
# Let's now come back to the standard level of message for the rest of the Tutorial -- in yhis case neither `DEBUG`
# nor `INFO` messages will be printed.

# %%
scp.set_loglevel(scp.WARNING)

scp.info_('this is an info message!')
scp.debug_('this is a debug message!')

# %% [markdown]
#
# ### Error handling
# If something goes wrong with during a cell execution,  a ``traceback`` is displayed.
#
# For instance, the object or method ``toto`` does not exist in the API, so an error (`ImportError`) is generated
# when trying to import this from the API.
#
# Here we catch the error with a conventional `try-except` structure

# %%
try:
    from spectrochempy import toto
except ImportError as e:
    scp.error_("OOPS, THAT'S AN IMPORT ERROR! : %s" % e)

# %% [markdown]
# The error will stop the execution if not catched.
#
# This is a basic behavior of python : one way to avoid stopping the execution without displaying a message is :

# %%
try:
    from spectrochempy import toto  # noqa: F811, F401
except Exception:
    pass
