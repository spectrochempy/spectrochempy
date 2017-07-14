

# register to dataset

from .writejdx import write_jdx

from ..dataset.api import NDDataset

setattr(NDDataset, 'write_jdx', write_jdx)

# make also the reader available for the API

__all__ = ['write_jdx']