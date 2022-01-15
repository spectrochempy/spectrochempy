import numpy as np
import spectrochempy as scp


# ------------------------------------------------------------
# Create fake data to be used by analysis routine for testing
# ------------------------------------------------------------


def _make_spectra_matrix(modelname, ampl, pos, width, ratio=None, asym=None):
    x = scp.Coord(np.linspace(6000.0, 1000.0, 4000), units="cm^-1", title="wavenumbers")
    s = []
    for arg in zip(modelname, ampl, pos, width, ratio, asym):
        model = getattr(scp, arg[0] + "model")()
        kwargs = {argname: arg[index + 1] for index, argname in enumerate(model.args)}
        s.append(model.f(x.data, **kwargs))

    st = np.vstack(s)
    st = scp.NDDataset(
        data=st, units="absorbance", title="absorbance", coordset=[range(len(st)), x]
    )

    return st


def _make_concentrations_matrix(*profiles):
    t = scp.Coord(np.linspace(0, 10, 50), units="hour", title="time")
    c = []
    for p in profiles:
        c.append(p(t.data))
    ct = np.vstack(c)
    ct = ct - ct.min()
    ct = ct / np.sum(ct, axis=0)
    ct = scp.NDDataset(data=ct, title="concentration", coordset=[range(len(ct)), t])

    return ct


def _generate_2D_spectra(concentrations, spectra):
    """
    Generate a fake 2D experimental spectra

    Parameters
    ----------
    concentrations : |NDDataset|
    spectra : |NDDataset|

    Returns
    -------
    |NDDataset|

    """
    from spectrochempy.core.dataset.npy import dot

    return dot(concentrations.T, spectra)


def generate_fake():
    # from spectrochempy.utils import show

    # define properties of the spectra and concentration profiles
    # ----------------------------------------------------------------------------------------------------------------------

    POS = (
        6000.0,
        4000.0,
    )  # 2000., 2500.)
    WIDTH = (6000.0, 1000.0)  # , 250., 800.)
    AMPL = (100.0, 70.0)  # , 10., 50.)
    RATIO = (0.0, 0.0)  # , .2, 1.)
    ASYM = (0.0, 0.0)  # , .2, -.6)

    MODEL = ("gaussian", "gaussian")  # , "asymmetricvoigt", "asymmetricvoigt")

    def C1(t):
        return t * 0.05 + 0.01  # linear evolution of the baseline

    def C2(t):
        return scp.sigmoidmodel().f(t, 0.5, max(t) / 2.0, -2)

    def C3(t):
        return np.exp(-t / 3.0) * 0.7

    def C4(t):
        return 1.0 - C2(t) - C3(t)

    spec = _make_spectra_matrix(MODEL, AMPL, POS, WIDTH, RATIO, ASYM)
    spec.plot()

    conc = _make_concentrations_matrix(C1, C2)  # , C3, C4)
    conc.plot(colorbar=False)

    d = _generate_2D_spectra(conc, spec)
    # add some noise
    d.data = np.random.normal(d.data, 0.002 * d.data.max())

    d.plot()


#    d.save('test_full2D')
#    spec.save('test_spectra')
#    conc.save('test_concentration')

if __name__ == "__main__":

    generate_fake()
    scp.show()
