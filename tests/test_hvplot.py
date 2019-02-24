from spectrochempy import *
import pytest

@pytest.mark.skip
def test_hvplot(IR_dataset_1D):

    nd = IR_dataset_1D
    nd.hvplot()

@pytest.mark.skip
def test2_hvplot(IR_dataset_2D):

    nd = IR_dataset_2D
    nd.y -= nd.y[0]
    nd.y.title = 'Time'

    nd.hvplot()