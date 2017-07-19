import sys
from traitlets import import_item

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.utils import list_packages

name = 'databases'
pkgs = sys.modules['spectrochempy.%s'%name]
api = sys.modules['spectrochempy.%s.api'%name]

pkgs = list_packages(pkgs)

__all__ = []

for pkg in pkgs:
    if pkg.endswith('api'):
        continue
    pkg = import_item(pkg)
    if not hasattr(pkg, '__all__'):
        continue
    a = getattr(pkg,'__all__')
    __all__ += a
    for item in a:
        #setattr(NDDataset, item, getattr(pkg, item))
        setattr(api, item, getattr(pkg, item))

