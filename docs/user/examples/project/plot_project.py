# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================
"""
Project creation
=======================================

In this example, we create a Project from scratch

"""
import  spectrochempy as scp

##############################################################################
# Let's assume we have three subproject to group in a single project

proj = scp.Project(

    # subprojects
    scp.Project(name='P350', label=r'$\mathrm{M_P}\,(623\,K)$'),
    scp.Project(name='A350', label=r'$\mathrm{M_A}\,(623\,K)$'),
    scp.Project(name='B350', label=r'$\mathrm{M_B}\,(623\,K)$'),

    # attributes
    name='project_1',
    label='main project',

)

assert proj.projects_names == ['A350', 'B350', 'P350']

##############################################################################
# Add for example two datasets to the ``A350`` subproject.

ir = scp.NDDataset([1.1, 2.2, 3.3], coords=[[1, 2, 3]])
print(ir)
tg = scp.NDDataset([1, 3, 4], coords=[[1, 2, 3]])
print(tg)
proj.A350['IR'] = ir
proj.A350['TG'] = tg

##############################################################################
# Members of the project or attributes are easily accesssed:

print(proj.A350)
print(proj)
print(proj.A350.label)
print(proj.A350.TG)

##############################################################################
# Save this project

proj.save()

##############################################################################
# RELOAD the project from disk as newproj

newproj = scp.Project.load('project_1')
print(newproj)

assert str(newproj) == str(proj)
assert newproj.A350.label == proj.A350.label


##############################################################################
# Now we add a script to the original proj

script_source = """
set_loglevel(INFO)
info_('samples contained in the project are:%s'%proj.projects_names)
"""

proj['print_info'] = scp.Script('print_info', script_source)
print(proj)
print('*******************************************')
##############################################################################
# save but do not change the original data

proj.save(overwrite_data=False)

##############################################################################
# RELOAD it

newproj = scp.Project.load('project_1')
print(newproj)

##############################################################################
# Execute a script

scp.run_script(newproj.print_info)

##############################################################################
# Another way to do the same thing is ith the following syntax (which may
# seem simpler

newproj.print_info()

###############################################################################
# Finally lets use a more usefull script
script_source_2 = """
proj.A350.TG.plot_scatter(title='my scatter plot')
#show()
"""
proj['tgscatter'] = scp.Script('tgscatter', script_source_2)

proj.tgscatter()

#show() # uncomment to show plot if needed()

