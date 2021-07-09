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
# ## API Configuration
#
# Many options of the API can be set up.
#Let's first import it in the u the usual way:

# %%
import spectrochempy as scp

# %% [markdown]
# ### General information
# General informations on the API can be obtained through the following
print(f'       scp.copyright : {scp.copyright}')
print(f'         scp.version : {scp.version}')
print(f'         scp.release : {scp.release}')
print(f'         scp.license : {scp.license}')
print(f'             scp.url : {scp.url}')
print(f'    scp.release_date : {scp.release_date}')
print(f'         scp.authors : {scp.authors}')
print(f'    scp.contributors : {scp.contributors}')
print(f'     scp.description : {scp.description}')
print(f'scp.long_description : {scp.long_description}')

# ### Loglevel
# During the execution, the API can display, besides the expected output, several types of information thant can be
# ranked by criticity:
#
# | Loglevel | criticity level | use |
# | --- | --- | --- |
# | `DEBUG`    | `10` | help diagnose problems on the running process or help developers |
# | `INFO`     | `20` | general information on the running process |
# | `WARNING`  | `30` | a condition might cause a problem with respect to the expected behaviour |
# | `ERROR`    | `40` | wrong argument/commands are used or bug |
# | `CRITICAL` | `50` | the process could lead to a system crash |
# | --- | --- | --- |
#
#  Not all these informations are always necessary and the level of information displayed by SpectroChemPy can be
#  tuned using the command `scp.set_loglevel(criticity)` with the rule that only informations having a
#  criticity larger than that passed to the `set_loglevel()` function will be shown.
#
#  For instance, the `DEBUG` level can be triggered by using one of the three equivalent instruictions s
#
# ```python
# scp.set_loglevel('DEBUG')
# scp.set_loglevel(scp.DEBUG)
# scp.set_loglevel(10)
# ```
# The current loglevel can be obtained
#
# ```python
# scp.get_loglevel()
# ```
# For instance the following code shows how to print and change the current loglevel
# %%
print(f'Default: {scp.get_loglevel()}')       # print the current loglevel
scp.set_loglevel(scp.INFO)                    # set loglevel to 'INFO'
print(f'New loglevel: {scp.get_loglevel()}')
scp.set_loglevel('WARNING')                   #reset to default
print(f'New loglevel: {scp.get_loglevel()}')

# %% [markdown]
# as seen above, the INFO message: 'changed default log_level to INFO' has been displayed when the `ÌNFO` level
# has been set, and no such a message is displayed when the `WARNING`level
#
# It is also possible to define these messages in scripts:
# %%
scp.info_('this is an info message!')
scp.debug_('this is a debug message!')

# %% [markdown]
# Only the info message is displayed, as expected.
#
# If we change the loglevl to ``DEBUG``, we should get the two messages

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
# ### Error handling

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


