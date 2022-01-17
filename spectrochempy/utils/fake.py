import numpy as np
import spectrochempy as scp

__all__ = ["generate_fake"]

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
    t = scp.LinearCoord(np.linspace(0, 10, 50), units="hour", title="time")
    c = []
    for p in profiles:
        c.append(p(t.data))
    ct = np.vstack(c)
    ct = ct - np.min(ct)
    if ct.shape[0] > 1:
        ct = ct / np.sum(ct, axis=0)
    ct = scp.NDDataset(data=ct, title="concentration", coordset=[range(len(ct)), t])

    return ct


def generate_fake():
    """
    Generate a fake 2D experimental spectra

    returns
    -------
    datasets:
        2D spectra, individual spectra and concentrations
    """

    # define properties of the spectra and concentration profiles
    # ----------------------------------------------------------------------------------------------------------------------
    from spectrochempy.core.dataset.npy import dot

    # data for four peaks (one very broad)
    POS = (6000.0, 4000.0, 2000.0, 2500.0)
    WIDTH = (6000.0, 1000.0, 250.0, 800.0)
    AMPL = (100.0, 70.0, 10.0, 50.0)
    RATIO = (0.1, 0.5, 0.2, 1.0)
    ASYM = (0.0, 0.0, 0, 4)
    MODEL = ("gaussian", "voigt", "voigt", "asymmetricvoigt")

    def C1(t):
        return t * 0.05 + 0.01  # linear evolution of the baseline

    def C2(t):
        return scp.sigmoidmodel().f(t, 1.0, max(t) / 2.0, 1, 2)

    def C3(t):
        return scp.sigmoidmodel().f(t, 1.0, max(t) / 5.0, 1, -2)

    def C4(t):
        return 1.0 - C2(t) - C3(t)

    specs = _make_spectra_matrix(MODEL, AMPL, POS, WIDTH, RATIO, ASYM)

    concs = _make_concentrations_matrix(C1, C2, C3, C4)

    # make 2D
    d = dot(concs.T, specs)

    # add some noise
    d.data = np.random.normal(d.data, 0.005 * d.data.max())

    # d.plot()
    return d, specs, concs


if __name__ == "__main__":

    pass
