# DEPRECATED

from spectrochempy.analysis.baseline.baseline import Baseline
from spectrochempy.application import warning_
from spectrochempy.utils.decorators import deprecated

__all__ = ["BaselineCorrection", "ab", "abc"]
__dataset_methods__ = ["ab", "abc"]


class BaselineCorrection(Baseline):
    @deprecated(replace="Baseline", removed="0.7.0")
    def __init__(self, dataset):
        super().__init__()

        self.dataset = dataset

    def __call__(self, *args, **kwargs):
        return self.compute(*args, **kwargs)

    def compute(self, *ranges, **kwargs):
        if isinstance(ranges[0], tuple) and len(ranges) == 1:
            ranges = ranges[0]
        self.ranges = ranges

        # BaselineCorrection._setup
        method = kwargs.pop("method", None)
        interpolation = kwargs.pop("interpolation", None)
        if interpolation == "polynomial":
            order = int(kwargs.get("order", 1))
        if method == "multivariate":
            npc = int(kwargs.pop("npc", 5))
        # zoompreview = kwargs.get("zoompreview", 1.0)
        # figsize = kwargs.get("figsize", (7, 5))

        # translate to new Baseline parameters
        self.multivariate = method == "multivariate"
        self.model = "polynomial"
        self.order = order if interpolation == "polynomial" else "pchip"
        self.n_components = npc if method == "multivariate" else 0

        self.fit(self.dataset)
        return self.transform()

    def run(self, *args, **kwargs):
        warning_(
            "Sorry but the interactive  method has been removed in version 0.6.6. "
            "Use BaselineCorrector instead"
        )
        return self.compute(*args, **kwargs)


@deprecated(replace="basc with order=1 and model='polynomial'", removed="0.7.0")
def ab(dataset, **kwargs):
    return abc(dataset, **kwargs)


@deprecated(replace="basc with order=1 and model='polynomial'", removed="0.7.0")
def abc(dataset, **kwargs):
    blc = BaselineCorrection(dataset)
    blc.model = "polynomial"
    blc.order = 1
    blc.fit(dataset)
    return blc.transform()
