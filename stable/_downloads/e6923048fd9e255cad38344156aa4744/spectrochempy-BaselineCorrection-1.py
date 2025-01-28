from spectrochempy import *
nd = NDDataset.read_omnic('irdata/nh4y-activation.spg')
ndp = nd[:, 1291.0:5999.0]
bc = BaselineCorrection(ndp)
ranges=[[5996., 5998.], [1290., 1300.],
        [2205., 2301.], [5380., 5979.],
        [3736., 5125.]]
span = bc.compute(*ranges,method='multivariate',
                  interpolation='pchip', npc=8)
_ = bc.corrected.plot_stack()
show()