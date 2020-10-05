
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

__all__=['colors']

class colorscale:

    def normalize(self, data, cmap='viridis'):
        _colormap = plt.get_cmap(cmap)
        amp = 0. # np.ma.ptp(data) / 10.
        range = [np.ma.min(np.ma.min(data) - amp), np.ma.max(np.ma.max(data)) + amp]
        range.sort()
        _norm = mpl.colors.Normalize(vmin=range[0], vmax=range[1])
        self.scalarMap = mpl.cm.ScalarMappable(norm=_norm, cmap=_colormap)

    def rgba(self, z):
            c = self.scalarMap.to_rgba(z)[0]
            c[0:3] *= 255
            c[0:3] = np.round(c[0:3].astype('uint16'),0)
            return  f'rgba'+str(tuple(c))

colorscale = colorscale()