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
#       jupytext_version: 1.10.2
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
# ## API Loading
import spectrochempy as scp

# %% [markdown]
# ## API Configuration
#
# Many options of the API can be set up

# %%
scp.set_loglevel(scp.INFO)

# %% [markdown]
# In the above cell, we have set the **log** level to display ``info`` messages, such as this one:

# %%
scp.info_('this is an info message!')
scp.debug_('this is a debug message!')

# %% [markdown]
# Only the info message is displayed, as expected.
#
# If we change it to ``DEBUG``, we should get the two messages

# %%
scp.set_loglevel(scp.DEBUG)

scp.info_('this is an info message!')
scp.debug_('this is a debug message!')

# %% [markdown]
# Let's now come back to a standard level of message for the rest of the Tutorial.

# %%
scp.set_loglevel(scp.WARNING)

scp.info_('this is an info message!')
scp.debug_('this is a debug message!')


# %% [markdown]
# Other configuration items will be further described when necessary in the other chapters.

# %% [markdown]
# ## Errors

# If something goes wrong with during a cell execution,  a ``traceback`` is displayed.
#
# For instance, the object or method ``toto`` does not exist in the API, so an error (**ImportError**) is generated
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
# This is a basic behavior of python : one way to avoid stoppping the execution without displaying a message is :

# %%
try:
    from spectrochempy import toto  # noqa: F811, F401
except Exception:
    pass


