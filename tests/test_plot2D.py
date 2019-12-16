# -*- coding: utf-8 -*-
#
# ======================================================================================================================

# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================



from spectrochempy import NDDataset, show, general_preferences as prefs


def test_plot2D():

    A = NDDataset.read_omnic('irdata/nh4y-activation.spg',
                             directory=prefs.datadir)
    A.y -= A.y[0]
    A.y.to('hour', inplace=True)
    A.y.title = u'Aquisition time'
    ax = A.copy().plot_stack()
    axT = A.copy().plot_stack(data_transposed=True)
    ax2 = A.copy().plot_image(style=['sans', 'paper'], fontsize=9)

    mystyle = {'image.cmap': 'magma',
               'font.size': 10,
               'font.weight': 'bold',
               'axes.grid': True}
    # TODO: store these styles for further use
    A.plot(style=mystyle)
    A.plot(style=['sans', 'paper', 'grayscale'], colorbar=False)
    show()

    pass






# ======================================================================================================================

if __name__ == '__main__':
    pass
