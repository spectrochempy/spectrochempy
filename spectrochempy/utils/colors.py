
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

__all__=['colors']

class colorscale:

    def normalize(self, data, cmap='viridis'):
        _colormap = plt.get_cmap(cmap)
        p = data.ptp()*0.02
        _norm = mpl.colors.Normalize(vmin=data.min()-p, vmax=data.max()+p)
        self.scalarMap = mpl.cm.ScalarMappable(norm=_norm, cmap=_colormap)

    def rgba(self, z):
            c = self.scalarMap.to_rgba(z)[0]
            c[0:3] *= 255
            c[0:3] = np.round(c[0:3].astype('uint16'),0)
            return  f'rgba'+str(tuple(c))

colorscale = colorscale()