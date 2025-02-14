# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
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
#   nbsphinx:
#     orphan: true
# ---

# %% [markdown]
# # Markdown Cheat Sheet

# %% [markdown]
# Copied and adapted
# from __[this guide](https://www.ibm.com/support/knowledgecenter/en/SSGNPV_2.0.0/dsx/markd-jupyter.html)__!
#
# This Markdown cheat sheet provides a quick overview of all the Markdown syntax elements to format Markdown cells in
# Jupyter notebooks.

# %% [markdown]
# ## Headings

# %% [markdown]
# Use the number sign (#) followed by a blank space for notebook titles and section headings, e.g.:
# ```md
# # for titles
# ## for major headings
# ### for subheadings
# #### for 4th level subheading
# ```

# %% [markdown]
# ## Emphasis

# %% [markdown]
# Use the surrounding _ or * to emphasize text, e.g.:
# ```
# Bold text: `__string___ or **string**`
# Italic text:  `_string_ or *string`
# ```

# %% [markdown]
# ## Mathematical symbols

# %% [markdown]
# Surround mathematical symbols with a dollar sign (\$), for example:
# ```
# $ \lambda = \sqrt{2*\pi} $
# ```
# gives $ \lambda = \sqrt{2*\pi} $

# %% [markdown]
# ## Monospace font

# %% [markdown]
# Surround text with a grave accent (\` ) also called a back single quotation mark, for example:
# ```
# `string`
# ```
# You can use the monospace font for `file paths` , `file names` ,`message text` ...

# %% [markdown]
# ## Line breaks

# %% [markdown]
# Sometimes markdown does not make line breaks when you want them. To force a linebreak, use the following code: `<br>`

# %% [markdown]
# ## Indenting

# %% [markdown]
# Use the greater than sign (>) followed by a space, for example:
# ```
# > Text that will be indented when the Markdown is rendered.
# Any subsequent text is indented until the next carriage return.
# ```

# %% [markdown]
# ## Bullets

# %% [markdown]
# To create a circular bullet point, use one of the following methods. Each bullet point must be on its own line.
#
# -  A hyphen (-) followed by one or two spaces, for example:
#
# ```
# - Bulleted item
# ```
#
#  - A space, a hyphen (-) and a space, for example:
#
# ```
#  - Bulleted item
# ```
#
# * An asterisk (*) followed by one or two spaces, for example:
#
# ```
# * Bulleted item
# ```
#
# To create a sub bullet, press Tab before entering the bullet point using one of the methods described above. For
# example:
#
# ```
# - Main bullet point
#      - Sub bullet point
# ```

# %% [markdown]
# ## Numbered lists

# %% [markdown]
# To create a numbered list, enter 1. followed by a space, for example:
# ```
# 1. Numbered item
# 1. Numbered item
# ```
# For simplicity, you use 1. before each entry. The list will be numbered correctly when you run the cell.
#
# To create a substep, press Tab before entering the numbered item, for example:
# ```
# 1. Numbered item
#      1. Substep
# ```

# %% [markdown]
# ## Colored note boxes

# %% [markdown]
# Use one of the following <div> tags to display text in a colored box.
#
# **Restriction**:
# Not all Markdown code displays correctly within <div> tags, so review your colored boxes carefully.
# For example, to make a word bold, surround it with the HTML code for bold
# (`<b>text</b>` -> <b>text</b>) instead of the Markdown code.
#
# The color of the box is determined by the alert type that you specify:
#
# * Blue boxes (alert-info)
# * Yellow boxes (alert-warning)
# * Green boxes (alert-success)
# * Red boxes (alert-danger)
#
# ```
# <div class="alert alert-block alert-info">
# <b>Tip:</b>  For example use blue boxes to highlight a tip.
# If it’s a note, you don’t have to include the word “Note”.
# </div>
# ```
#
# <div class="alert alert-block alert-info">
# <b>Tip:</b> For example use blue boxes to highlight a tip.
# If it’s a note, you don’t have to include the word “Note”.
# </div>

# %% [markdown]
# ## Graphics

# %% [markdown]
# You can attach image files directly to a notebook in Markdown cells by dragging and dropping it into the cell.
# To add images to other types of cells, you must use a graphic that is hosted on the web and use the following code
# to insert the graphic:
# ```
# <img src="url.gif" alt="Alt text that describes the graphic" title="Title text" />
#
# ```
# <img src="images/scpy.png" alt="Alt text that describes the graphic" width=100 title="Title text" />
#
# **Restriction**
# You cannot add captions to graphics.

# %% [markdown]
# ## Geometric shapes
# Use &# followed by the decimal or hex reference number for the shape, for example:
# ```
# &#reference_number;
# ```
# e.g., `&#9664;`: &#9664;
#
# For a list of reference numbers, see __[UTF-8 Geometric shapes](https://en.wikipedia.org/wiki/Geometric_Shapes)__.

# %% [markdown]
# ## Horizontal lines
# On a new line, enter three asterisks: `***`
# ***

# %% [markdown]
# ## Internal links
# To link to a section within your notebook, use the following code:
# ```
# [Section title](#section-title)
# ```
#
# For the text inside the parentheses, replace any spaces and special characters with a hyphen. For example,
# if your section is called `Processing functions` , you'd enter:
# ```
# [processing functions](#processing-functions)
# ```
# [processing functions](#processing-functions)

# %% [markdown]
# ## Processing functions
# This is the section that the internal link points to.

# %% [markdown]
# Now you can link to the section using:
# ```
# [processing functions](#processing-functions)
# ```
# [processing functions](#processing-functions)

# %% [markdown]
# ## External links

# %% [markdown]
# To link to an external site, use the following code:
# ```
#  __[link text](https://github.com/spectrochempy/spectrochempy)__
# ```
# Surround the link with two underscores (_) on each side
# __[link text](https://github.com/spectrochempy/spectrochempy)__
