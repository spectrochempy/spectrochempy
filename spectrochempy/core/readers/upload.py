# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

"""
In this module, methods are provided to download external datasets
from public database.

"""
__all__ = ['upload_IRIS']

# ----------------------------------------------------------------------------------------------------------------------
# standard imports
# ----------------------------------------------------------------------------------------------------------------------

from io import StringIO

# ----------------------------------------------------------------------------------------------------------------------
# third party imports
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
import requests

# ----------------------------------------------------------------------------------------------------------------------
# localimports
# ----------------------------------------------------------------------------------------------------------------------

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.dataset.ndcoord import Coord


# ............................................................................
def upload_IRIS():
    """
    Upload the classical IRIS dataset from the UCI distant repository

    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

    try:
        connection = True
        response = requests.get(url, stream=True, timeout=10)
    except requests.ConnectTimeout:
        connection = False

    if connection:  # Download data
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

        coordx = Coord(labels=['sepal_length', 'sepal width', 'petal_length', 'petal_width'], title='features')
        coordy = Coord(labels=labels, title='samples')

        new = NDDataset(data,
                        coords=[coordy, coordx],
                        title='size',
                        name='IRIS Dataset',
                        units='cm')

        history = 'Loaded from UC Irvine machine learning repository'

        return new

    else:
        # Cannot download - use the scikit-learn dataset (if scikit-learn is installed)
        from spectrochempy.core import HAS_SCIKITLEARN

        if HAS_SCIKITLEARN:
            from sklearn import datasets

            # import some data to play with
            data = datasets.load_iris()

            coordx = Coord(labels=['sepal_length', 'sepal width', 'petal_length', 'petal_width'], title='features')
            labels = [data.target_names[i] for i in data.target]
            coordy = Coord(labels=labels, title='samples')

            new = NDDataset(data.data,
                            coords=[coordy, coordx],
                            title='size',
                            name='IRIS Dataset',
                            units='cm')

            history = 'Loaded from scikit-learn datasets'

            return new

        else:

            raise IOError('Failed in uploading the IRIS dataset!')


# make a NDDataset class method
NDDataset.upload_IRIS = upload_IRIS

# ======================================================================================================================
if __name__ == '__main__':
    pass
