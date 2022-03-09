import re
from functools import partial

from ipywidgets import widgets, Layout
from IPython.display import display

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.plotters.multiplot import multiplot
from spectrochempy.core.processors.baseline import BaselineCorrection
from spectrochempy.core.processors.concatenate import concatenate
from spectrochempy.core.readers.importer import read
from spectrochempy.utils import show
from spectrochempy.core import warning_, info_

__all__ = ["BaselineCorrector"]


class BaselineCorrector:
    """
    Launch a GUI for baseline corrections.

    Wrapper of scp.BaselineCorrection(X), with widgets for dataset slicing,
    input parameters and graphical output.
    Should be run in jupyter notebook (does not always run properly in jupyter
     lab) with the widget backend (magic `%matplotlib widget`).

    Parameters
    ----------
    X : |NDDataset|, default: None
        The NDDataset to process. If None, an upload button can be used to load data.
    initial_ranges : list, optional, default: None
        The initial regions where to compute the baseline. If not given, 5% on each
        side of the spectra will be taken as a starting range's list.

    Attributes
    ----------
    original : NDDataset
        The (sliced) NDDataset, without baseline correction
    corrected : NDDataset
        The baseline corrected NDDataset
    baseline : NDDataset
        The baselines

    See Also
    --------
    BaselineCorrection : Baseline correction method.

    Notes
    -----
    The `BaselineCorrector()` widget can be used in jupyter notebook and jupyter lab.
    - The GUI buttons are the following:
        - `upload`: allows uploading files.
          Uploading file will trigger the reading and processing with default parameters
          (linear baseline with ranges of 5% of the x axis at both ends). If a NDDataset
          has been passed or a file has been previously loaded, BaselineCorrector will
          be reset with the new file(s).
        - `process`: triggers baseline correct and plotting of original
          dataset + baseline and corrected datasets
        - `save as`: save the baseline corrected dataset
    - The `x slice` and `y slice` textboxes can be used to slice the initial
      dataset with the usual `[start:stop:step]`
      format. In both dimensions, coordinates or indexes
      can be used (for example, [3000.0::2] or [:100:5] are valid entries).
    - `Method` and `Interpolation` dropdown fields are self explaining,
      see BaselineCorrection() for details.
    - Ranges should be entered as a series of intervals or wavenumbers, e.g.
      ```
      [5900.0, 5400.0],
      2000.0,
      [1550.0, 1555.0],
      ```

    Examples
    --------
    # runs only in Jupyter notebook/ipython environment.
    # See : User's Guide > Processing > Baseline corrections for an example.
    [In 2] X = scp.read_omnic("irdata/nh4y-activation.spg")
           out = scp.BaselineCorrector(X)
    """

    def __init__(self, X=None, initial_ranges=None):

        if not isinstance(X, (NDDataset, type(None))):
            raise ValueError("X must be None or a valid NDDataset")

        self._X = X
        self._fig = None
        self._initial_ranges = initial_ranges
        self._done = False

        # I/O and processing widgets
        self._load_button = widgets.Button(description="Upload", icon="upload")
        self._load_button.on_click(self._load_clicked)
        self._process_button = widgets.Button(description="Process", icon="play")
        self._process_button.on_click(self._process_clicked)
        self._save_button = widgets.Button(description="Save as", icon="save")
        self._save_button.on_click(self._save_clicked)

        # Parameters widgets
        self._npc_slider = widgets.IntSlider(description="N pc", value=1, min=1, max=5)
        self._order_slider = widgets.IntSlider(
            description="Order", layout=Layout(width="350px"), value=1, min=1, max=6
        )

        self._method_selector = widgets.Dropdown(
            description="Method",
            options=["sequential", "multivariate"],
            value="sequential",
        )
        self._method_control = widgets.HBox(children=[self._method_selector])

        self._interpolation_selector = widgets.Dropdown(
            description="Interpolation",
            options=["polynomial", "pchip"],
            value="polynomial",
        )
        self._interpolation_control = widgets.HBox(
            children=[self._interpolation_selector, self._order_slider]
        )

        self._ranges_control = widgets.Textarea(description="Ranges:")
        self._x_limits_control = widgets.Text(description="x slice:")
        self._y_limits_control = widgets.Text(description="y slice:")
        self._limits_control = widgets.HBox(
            children=[self._x_limits_control, self._y_limits_control]
        )

        self._io = widgets.VBox()
        self._controls = widgets.VBox()

        self._input = widgets.HBox(children=[self._io, self._controls])
        self._output = widgets.Output()
        display(self._input)
        display(self._output)

        # init attributes
        self.original = NDDataset()
        self.corrected = NDDataset()
        self.baseline = NDDataset()

        # events
        for control in [
            "x_limits_control",
            "y_limits_control",
            "ranges_control",
            "npc_slider",
            "order_slider",
            "method_selector",
            "interpolation_selector",
        ]:
            getattr(self, f"_{control}").observe(
                partial(self._update, control=control), names="value"
            )

        # Start
        if self._X is not None:
            self._enabled_process(True)
            self._process_clicked()
        else:
            self._enabled_process(False)

    def _enabled_process(self, flag):
        if flag:
            self._io.children = [
                self._load_button,
                self._process_button,
                self._save_button,
            ]
            self._controls.children = [
                self._limits_control,
                self._method_control,
                self._interpolation_control,
                self._ranges_control,
            ]

        else:
            self._io.children = [self._load_button]
            self._controls.children = []
            with self._output:
                info_(
                    "No data have been defined.\n"
                    "Use the upload button to load data to be processed!."
                )

    def _update(self, *args, **kwargs):

        control = kwargs.get("control")
        if control == "method_selector":
            self._method_control.children = (self._method_selector,)
            if self._method_selector.value == "multivariate":
                self._method_control.children = (
                    self._method_selector,
                    self._npc_slider,
                )

        elif control == "interpolation_selector":
            self._interpolation_control.children = (self._interpolation_selector,)
            if self._interpolation_selector.value == "polynomial":
                self._interpolation_control.children = (
                    self._interpolation_selector,
                    self._order_slider,
                )

        self._process_button.disabled = False  # probably need a reprocessing

    def _blcorrect_and_plot(self):
        slice_x = _str_to_slice(self._x_limits_control.value, self._X, "x")
        slice_y = _str_to_slice(self._y_limits_control.value, self._X, "y")
        self.original = self._X[slice_y, slice_x]
        ranges = _str_to_ranges(self._ranges_control.value)

        if self.original is not None:  # slicing was OK

            # check that no range is outside coordinates
            new_ranges, changed = _update_ranges(
                _str_to_ranges(self._ranges_control.value), self.original.x.data
            )
            if changed:
                ranges = _round_ranges(new_ranges)
                self._ranges_control.value = _ranges_to_str(ranges)

            blc = BaselineCorrection(self.original)
            self.corrected = blc.compute(
                *ranges,
                interpolation=self._interpolation_selector.value,
                order=self._order_slider.value,
                method=self._method_selector.value,
                npc=self._npc_slider.value,
            )
            self.baseline = self.original - self.corrected

            self._output.clear_output(True)
            with self._output:
                axes = multiplot(
                    [
                        concatenate(self.original, self.baseline, dims="y"),
                        self.corrected,
                    ],
                    labels=["Original", "Corrected"],
                    sharex=True,
                    nrow=2,
                    ncol=1,
                    fig=self._fig,
                    figsize=(8, 6),
                    dpi=96,
                    left=0.12,
                    mpl_event=False,
                )
                axes["axe11"].get_xaxis().set_visible(False)
                blc.show_regions(axes["axe21"])
                self._fig = axes["axe21"].figure
                show()
            self._done = True

    def _load_clicked(self, b=None):
        # read data and reset defaults
        ds = read()
        if ds is not None:
            if isinstance(ds, NDDataset):
                self._X = ds
                self._method_selector.value = "sequential"
                self._interpolation_selector.value = "polynomial"
                self._order_slider.value = 1
                self._npc_slider.value = 1
                self._done = False
                self._enabled_process(True)
                self._process_clicked()
            else:
                raise IOError("Could not read or merge uploaded files")
        else:
            self._enabled_process(False)
            warning_("process canceled because X is None")

    def _process_clicked(self, b=None):
        """(re)process dataset (slicing) and baseline correct"""

        if not self._done:
            # first processing,
            # defines default ranges (5% of the X axis at both ends)...
            len_ = int(len(self._X.x) / 20)
            if self._initial_ranges:
                ranges = self._initial_ranges
            else:
                ranges = _round_ranges(
                    (
                        [self._X.x.data[0], self._X.x.data[len_]],
                        [self._X.x.data[-len_], self._X.x.data[-1]],
                    )
                )
            self._ranges_control.value = _ranges_to_str(ranges)

            self._x_limits_control.value = _x_slice_to_str(
                slice(0, len(self._X.x), 1), self._X
            )
            self._y_limits_control.value = _y_slice_to_str(slice(0, len(self._X.y), 1))
            # ... and baseline correct with defaults

        self._blcorrect_and_plot()
        self._process_button.disabled = True

    def _save_clicked(self, b=None):
        if self.corrected is not None:
            return self.corrected.write()


# Utility functions
def _x_slice_to_str(slice, dataset, decimals=2):
    return (
        f"[{round(dataset.x.data[slice.start], decimals)} : "
        f"{round(dataset.x.data[slice.stop - 1], decimals)} : {slice.step}]"
    )


def _y_slice_to_str(slice):
    return f"[{slice.start} : {slice.stop} : {slice.step}]"


def _str_to_num(strg):
    num = None
    if strg:
        num = int(strg) if "." not in strg else float(strg)
    return num


def _str_to_slice(strg, dataset, dim):
    regex = r"^\[?(\d*\.?\d*)\:?(\d*\.?\d*)\:?(-?\d*)\]?$"
    strg = strg.strip()
    strg = strg.replace(" ", "")
    match = re.search(regex, strg)
    if match:
        start = _str_to_num(match.group(1))
        stop = _str_to_num(match.group(2))
        step = _str_to_num(match.group(3))
        return dataset._get_slice(slice(start, stop, step), dim)
    else:
        raise ValueError(f"Something is wrong in the slice definition: {strg}.")


def _ranges_to_str(ranges):
    return str(ranges).replace("(", "").replace("], ", "],\n").replace(")", "")


def _str_to_ranges(strg):
    strg = strg.strip()
    strg = strg.replace(" ", "")
    return eval(f"({strg})")


def _round_ranges(ranges, decimals=2):
    """round x values for better reading"""
    ranges = list(ranges)
    for i, item in enumerate(ranges):
        if isinstance(item, float):  # pragma: cover
            # can not be reached in the current implementation of baselinecorrector
            ranges[i] = round(item, decimals)
        else:
            for j, sub_item in enumerate(item):
                ranges[i][j] = round(sub_item, decimals)
    return tuple(ranges)


def _update_ranges(ranges, coord, decimals=2):
    """return valid ranges

    Ranges outside the coord limits (if any), are replaced
    by the min or max the coords. Ranges with the limits or including
    the limits are unchanged.
    returns:
    --------
     list of ranges
     Bool True if changed
    """
    ranges = list(ranges)
    changed = False
    # sort coord id needed
    if coord[-1] < coord[0]:
        coord = coord[::-1]
    for i, item in enumerate(ranges):
        # if out of range, makes it within coord limits"
        if isinstance(item, float):
            if item < coord[0]:
                ranges[i] = coord[1]
                changed = True
            elif item > coord[-1]:
                ranges[i] = coord[-2]
                changed = True
        else:
            if max(item) < coord[0]:
                ranges[i] = [coord[0], coord[1]]
                changed = True
            if min(item) > coord[-1]:
                ranges[i] = [coord[-2], coord[-1]]
                changed = True
    return tuple(ranges), changed
