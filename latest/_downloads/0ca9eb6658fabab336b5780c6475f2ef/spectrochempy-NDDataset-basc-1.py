import spectrochempy as scp
nd = scp.read('irdata/nh4y-activation.spg')
ndp = nd[:, 1291.0:5999.0]

ranges=[[5996., 5998.], [1290., 1300.],
        [2205., 2301.], [5380., 5979.],
        [3736., 5125.]]

ndcorr = scp.basc(ndp, *ranges,method='multivariate', interpolation='pchip', npc=8)
ndcorr.plot()
scp.show()