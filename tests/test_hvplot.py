from spectrochempy import *


def test_hvplot(IR_dataset_1D):

    nd = IR_dataset_1D
    nd.hvplot()

def test2_hvplot(IR_dataset_2D):

    nd = IR_dataset_2D
    nd.y -= nd.y[0]
    nd.y.title = 'Time'

    nd.hvplot()