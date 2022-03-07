import re

import ipywidgets as widgets
from IPython.display import display

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.plotters.multiplot import multiplot
from spectrochempy.core.processors.baseline import BaselineCorrection
from spectrochempy.core.processors.concatenate import concatenate
from spectrochempy.core.readers.importer import read

from spectrochempy.core import warning_

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
    X : NDDataset, default: None
        The NDDataset to process. If None, a FileUpload widget is enabled.
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
    The `BaselineCorrector()` widget can be used in jupyter notebook (does not always run
    properly in jupyter lab).
    - The GUI buttons are as follows:
        - `upload`: upload files.
          Uploading file will trigger the reading and processing with default parameters
          (linear baseline with ranges of 5% of the x axis at both ends). If a NDDataset
          has been passed or a file has been previously loaded, BaselineCorrector will
          be reset with the new file(s).
        - `process`: triggers baseline correct and plotting of original
          dataset + baseline and corrected datasets
        - `save as`: save the baseline corrected NDDataset
    - The `x slice` and `y slice` textboxes can be used to slice the initial
      dataset with the usual `[start:stop:step]`
      format. Coordinates or indexes can be used
      (e.g. `[3000.0:2000.0:1]` or `[0:100:1]` are valid
      entries).
    - Method and Interpolation are self explaining, see BaselineCorrection() for
      details.
    - Ranges should be entered as a tuple of intervals or wavenumbers, e.g.
      ```
      (
      [5900.0, 5400.0],
      2000.0,
      [1550.0, 1555.0],
      )
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

        self._loadbutton = widgets.Button(description="upload", icon="upload")
        self._loadbutton.on_click(self.load_clicked)

        self._processbutton = widgets.Button(description="process", icon="play")
        self._processbutton.on_click(self.process_clicked)

        self._savebutton = widgets.Button(description="save as", icon="save")
        self._savebutton.on_click(self.save_clicked)

        self._methodselector = widgets.Dropdown(
            description="Method",
            options=["sequential", "multivariate"],
            value="sequential",
        )

        self._npcslider = widgets.IntSlider(description="npc", value=1, min=1, max=5)

        self._method_control = widgets.HBox(children=[self._methodselector])

        def toggle_npc(change):
            if self._methodselector.value == "multivariate":
                self._method_control.children = [self._methodselector, self._npcslider]
            else:
                self._method_control.children = [self._methodselector]

        self._methodselector.observe(toggle_npc, names="value")

        self._interpolationselector = widgets.Dropdown(
            description="Interpolation",
            options=["polynomial", "pchip"],
            value="polynomial",
        )

        self._orderslider = widgets.IntSlider(
            description="orderslider", value=1, min=1, max=6
        )

        self._interpolation_control = widgets.HBox(
            children=[self._interpolationselector, self._orderslider]
        )

        def toggle_polyorder(change):
            if self._interpolationselector.value == "polynomial":
                self._interpolation_control.children = [
                    self._interpolationselector,
                    self._orderslider,
                ]
            else:
                self._interpolation_control.children = [self._interpolationselector]

        self._interpolationselector.observe(toggle_polyorder, names="value")

        self._ranges_control = widgets.Textarea(description="Ranges:")
        self._x_limits_control = widgets.Text(description="x slice:")
        self._y_limits_control = widgets.Text(description="y slice:")
        self._limits_control = widgets.HBox(
            children=[self._x_limits_control, self._y_limits_control]
        )

        io = widgets.VBox(
            children=[
                self._loadbutton,
                self._processbutton,
                self._savebutton,
            ]
        )
        controls = widgets.VBox(
            children=[
                self._limits_control,
                self._method_control,
                self._interpolation_control,
                self._ranges_control,
            ]
        )

        self._input = widgets.HBox(children=[io, controls])
        self._output = widgets.Output(layout={"border": ".1px solid black"})
        display(self._input, self._output)

        if self._X is not None:
            self.process_clicked()
        else:
            with self._output:
                warning_(
                    "No data have been defined.\n"
                    "Use the upload button to load data to be processed!."
                )
            self.corrected = NDDataset()

    def blcorrect_and_plot(self):
        slice_x = _str_to_slice(self._x_limits_control.value.strip(), self._X, "x")
        slice_y = _str_to_slice(self._y_limits_control.value.strip(), self._X, "y")
        self.original = self._X[slice_y, slice_x]
        ranges = eval(self._ranges_control.value.strip())
        if self.original is not None:  # slicing was OK

            # check that no range is outside coordinates
            new_ranges, changed = _update_ranges(
                eval(self._ranges_control.value.replace(" ", "")), self.original.x.data
            )
            if changed:
                ranges = _round_ranges(new_ranges)
                self._ranges_control.value = (
                    str(ranges)
                    .replace("(", "(\n")
                    .replace("], ", "],\n")
                    .replace(")", "\n)")
                )

            blc = BaselineCorrection(self.original)
            self.corrected = blc.compute(
                *ranges,
                interpolation=self._interpolationselector.value,
                order=self._orderslider.value,
                method=self._methodselector.value,
                npc=self._npcslider.value,
            )
            self.baseline = self.original - self.corrected

            self._output.clear_output()
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
                    figsize=(7, 6),
                    dpi=96,
                    mpl_event=False,
                )
                axes["axe11"].get_xaxis().set_visible(False)
                blc.show_regions(axes["axe21"])
                self._fig = axes["axe21"].figure
                self._fig.show()
            self._done = True

    def load_clicked(self, b=None):
        # read data and reset defaults
        ds = read()
        if ds is not None:
            if isinstance(ds, NDDataset):
                self._X = ds
                self._methodselector.value = "sequential"
                self._interpolationselector.value = "polynomial"
                self._orderslider.value = 1
                self._npcslider.value = 1
                self._done = False
                self.process_clicked()
            else:
                raise IOError("Could not read or merge uploaded files")
        else:
            warning_("process canceled because X is None")

    def process_clicked(self, b=None):
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
            self._ranges_control.value = (
                str(ranges)
                .replace("(", "(\n")
                .replace("], ", "],\n")
                .replace(")", "\n)")
            )
            self._x_limits_control.value = _x_slice_to_str(
                slice(0, len(self._X.x), 1), self._X
            )
            self._y_limits_control.value = _y_slice_to_str(slice(0, len(self._X.y), 1))
            # ... and baseline correct with defaults

        self.blcorrect_and_plot()

    def save_clicked(self, b=None):
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
    strg = strg.replace(" ", "")
    match = re.search(regex, strg)
    if match:
        start = _str_to_num(match.group(1))
        stop = _str_to_num(match.group(2))
        step = _str_to_num(match.group(3))
        return dataset._get_slice(slice(start, stop, step), dim)
    else:
        raise ValueError(f"Something is wrong in the slice definition: {strg}.")


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
