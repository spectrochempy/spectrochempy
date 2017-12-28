# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

"""
In this module, methods are provided to download external datasets
from public database.

"""
__all__ = ['upload_IRIS']

# ----------------------------------------------------------------------------
# standard imports
# ----------------------------------------------------------------------------

from io import StringIO

# ----------------------------------------------------------------------------
# third party imports
# ----------------------------------------------------------------------------

import numpy as np
import requests

# ----------------------------------------------------------------------------
# localimports
# ----------------------------------------------------------------------------

from spectrochempy.dataset.nddataset import NDDataset
from spectrochempy.dataset.ndcoords import Coord
from spectrochempy.dataset.ndplot import show


# ............................................................................
def upload_IRIS():
    """
    Upload the classical IRIS dataset from the UCI distant repository

    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases" \
              "/iris/iris.data"

    response = requests.get(url, stream=True)

    txtdata = ''
    for rd in response.iter_content():
            txtdata += rd.decode('utf8')

    fil = StringIO(txtdata)
    try:
        data = np.loadtxt(fil, delimiter=',', usecols=range(4))
        fil.seek(0)
        labels = np.loadtxt(fil, delimiter=',', usecols=(4,), dtype='|S')
        labels = list((l.decode("utf8") for l in labels))
    except:
        raise IOError(
            '{} is not a .csv file or its structure cannot be recognized')

    coord0 = Coord(labels=labels, title='samples')
    coord1 = Coord(labels = ['sepal_length', 'sepal width', 'petal_length',
                             'petal_width'], title='features')

    new = NDDataset(data,
                    coordset=[coord0, coord1],
                    title='size',
                    name='IRIS Dataset',
                    units='cm')
    history = 'loaded from UC Irvine machine learning repository'

    return new

# make a NDDataset class method
NDDataset.upload_IRIS = upload_IRIS


# =============================================================================
if __name__ == '__main__':

    from spectrochempy.scp import PCA

    ds = upload_IRIS()
    print (ds)
    ds.plot_stack()

    pca = PCA(ds, centered=True)
    L, S = pca.transform()
    L.plot_stack()
    pca.screeplot()
    pca.scoreplot(1,2, color_mapping='labels')
    show()