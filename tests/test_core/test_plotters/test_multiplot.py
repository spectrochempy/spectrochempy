# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import os

from spectrochempy.core import preferences
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.plotters.multiplot import (
    multiplot,
    multiplot_map,
    multiplot_stack,
)
from spectrochempy.utils.plots import show

prefs = preferences


def test_multiplot():
    dataset = NDDataset.read_omnic(
        os.path.join(prefs.datadir, "irdata", "nh4y-activation.spg")
    )[:, 0:100]

    datasets = [dataset, dataset * 1.1, dataset * 1.2, dataset * 1.3]
    labels = ["sample {}".format(label) for label in ["1", "2", "3", "4"]]
    multiplot(
        datasets=datasets,
        method="stack",
        labels=labels,
        nrow=2,
        ncol=2,
        figsize=(9, 5),
        style="sans",
        sharex=True,
        sharey=True,
        sharez=True,
    )

    multiplot(
        datasets=datasets,
        method="image",
        labels=labels,
        nrow=2,
        ncol=2,
        figsize=(9, 5),
        sharex=True,
        sharey=True,
        sharez=True,
    )

    datasets = [
        dataset * 1.2,
        dataset * 1.3,
        dataset,
        dataset * 1.1,
        dataset * 1.2,
        dataset * 1.3,
    ]
    labels = ["sample {}".format(label) for label in ["1", "2", "3", "4", "5", "6"]]
    multiplot_map(
        datasets=datasets,
        labels=labels,
        nrow=2,
        ncol=3,
        figsize=(9, 5),
        sharex=False,
        sharey=False,
        sharez=True,
    )

    multiplot_map(
        datasets=datasets,
        labels=labels,
        nrow=2,
        ncol=3,
        figsize=(9, 5),
        sharex=True,
        sharey=True,
        sharez=True,
    )

    datasets = [
        dataset * 1.2,
        dataset * 1.3,
        dataset,
    ]
    labels = ["sample {}".format(label) for label in ["1", "2", "3"]]
    multiplot_stack(
        datasets=datasets,
        labels=labels,
        nrow=1,
        ncol=3,
        figsize=(9, 5),
        sharex=True,
        sharey=True,
        sharez=True,
    )

    multiplot_stack(
        datasets=datasets,
        labels=labels,
        nrow=3,
        ncol=1,
        figsize=(9, 5),
        sharex=True,
        sharey=True,
        sharez=True,
    )

    multiplot(
        method="pen",
        datasets=[dataset[0], dataset[10] * 1.1, dataset[19] * 1.2, dataset[15] * 1.3],
        nrow=2,
        ncol=2,
        figsize=(9, 5),
        labels=labels,
        sharex=True,
    )

    show()
