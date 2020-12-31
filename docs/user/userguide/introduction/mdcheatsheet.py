# -*- coding: utf-8 -*-
# # Markdown Cheat Sheet
#
# Copied and adapted from [this guide](https://www.ibm.com/support/knowledgecenter/en/SSGNPV_2.0.0/dsx/markd-jupyter
# .html)!
#
# This Markdown cheat sheet provides a quick overview of all the Markdown syntax elements to format Markdown cells in
# Jupyter notebooks.
#
# ## Headings
# ***
#
# Use the number sign (#) followed by a blank space for notebook titles and section headings, e.g.:
# ```
# # for titles
# ## for major headings
# ### for subheadings
# #### for 4th level subheading
# ```
#
# ## Emphasis
# ***
#
# Use the surroundig _ or * to emphasize text, e.g.:
# ```
# Bold text: `__string___ or **string**`
# Italic text:  `_string_ or *string`
# ```
#
# ## Mathematical symbols
# ***
#
# Surround mathematical symbols with a dollar sign (\$), for example:
# ```
# $ mathematical symbols $
# ```

# ## Monospace font
# ***
#
# Surround text with a grave accent (\`) also called a back single quotation mark, for example:
#
# ```
# `string`
# ```
#
# You can use the monospace font for file paths, file names, message text that users see, or text that users enter.
#
# ## Line breaks
# ***
# Sometimes markdown doesn’t make line breaks when you want them. To force a linebreak, use the following code: `<br>`
#
# ## Indenting
# ***
# Use the greater than sign (>) followed by a space, for example:
#
# ```
# > Text that will be indented when the Markdown is rendered.
# Any subsequent text is indented until the next carriage return.
# ```
#
# ## Bullets
# ***
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
# ````
#
# To create a sub bullet, press Tab before entering the bullet point using one of the methods described above. For
# example:
#
# ```
# - Main bullet point
#      - Sub bullet point
# ```
#
# ## Numbered lists
# To create a numbered list, enter 1. followed by a space, for example:
# 1. Numbered item
# 1. Numbered item
# For simplicity, you use 1. before each entry. The list will be numbered correctly when you run the cell.
# To create a substep, press Tab before entering the numbered item, for example:
# 1. Numbered item
#      1. Substep
# ## Colored note boxes
# Use one of the following <div> tags to display text in a colored box.
# Restriction
# Not all Markdown code displays correctly within <div> tags, so review your colored boxes carefully.
# For example, to make a word bold, surround it with the HTML code for bold (<b>text</b> instead of the Markdown code.
# The color of the box is determined by the alert type that you specify:
# Blue boxes (alert-info)
# <div class="alert alert-block alert-info">
# <b>Tip:</b> Use blue boxes (alert-info) for tips and notes.
# If it’s a note, you don’t have to include the word “Note”.
# </div>
# Yellow boxes (alert-warning)
# <div class="alert alert-block alert-warning">
# <b>Example:</b> Use yellow boxes for examples that are not
# inside code cells, or use for mathematical formulas if needed.
# </div>
# Green boxes (alert-success)
# <div class="alert alert-block alert-success">
# <b>Up to you:</b> Use green boxes sparingly, and only for some specific
# purpose that the other boxes can't cover. For example, if you have a lot
# of related content to link to, maybe you decide to use green boxes for
# related links from each section of a notebook.
# </div>
# Red boxes (alert-danger)
# <div class="alert alert-block alert-danger">
# <b>Just don't:</b> In general, avoid the red boxes. These should only be
# used for actions that might cause data loss or another major issue.
# </div>
#
# ## Graphics
# You can attach image files directly to a notebook in Markdown cells by dragging and dropping it into the cell.
# To add images to other types of cells, you must use a graphic that is hosted on the web and use the following code
# to insert the graphic:
# <img src="url.gif" alt="Alt text that describes the graphic" title="Title text" />
# Restriction
# You cannot add captions to graphics.
# Geometric shapes
# Use &# followed by the decimal or hex reference number for the shape, for example:
# &#reference_number
# For a list of reference numbers, see UTF-8 Geometric shapes.
# Horizontal lines
# On a new line, enter three asterisks:
# ***
# Internal links
# To link to a section within your notebook, use the following code:
# [Section title](#section-title)
# For the text inside the parentheses, replace any spaces and special characters with a hyphen. For example,
# if your section is called Analyzing customer purchasing habits, you'd enter:
# [Analyzing customer purchasing habits](#analyzing-customer-purchasing-habits)
# Alternatively, you can add an ID above the section:
# <a id="section_ID"></a>
# Important
# Each ID in the notebook must be unique.
# To link to a section that has an ID, use the following code:
# [Section title](#section_ID)
# Important
# Test all internal links to ensure that they work.
# External links
# To link to an external site, use the following code:
# __[link text](http://url)__
# Surround the link with two underscores (_) on each side
# Important
# Test all links to ensure that they work.
