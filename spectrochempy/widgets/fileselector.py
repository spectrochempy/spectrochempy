# ----------------------------------------------------------------------------------------------------------------------
# Modified from intake.gui
#
# Copyright (c) 2012 - 2018, Anaconda, Inc. and Intake contributors
# All rights reserved.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------------------------------------------------

"""
Widgets for the Jupyter noteboobk and Jupyter lab

"""
from IPython.core.interactiveshell import InteractiveShell
import IPython
from contextlib import contextmanager
import os

__all__ = ['FileSelector', 'URLSelector']

try:

    import ipywidgets as widgets

except ImportError:

    class FileSelector(object):
        def __repr__(self):
            raise RuntimeError("Please install ipywidgets to use the FileSe"
                               "lector")


@contextmanager
def ignore(ob):
    try:
        ob.ignore = True
        yield
    finally:
        ob.ignore = False


class Base(object):
    done_callback = None

    def stop(self, ok=False):
        done = self.done_callback is not None
        if ok and done:
            self.done_callback(ok)
        elif done:
            self.done_callback(None)

    def __repr__(self):
        return ("To get widget to display, you must "
                "install ipy/jupyter-widgets, run in a notebook and, in "
                "the case of jupyter-lab, install the jlab extension.")

    def _ipython_display_(self, **kwargs):

        # from IPython.Widget._ipython_display_
        if InteractiveShell.initialized():
            if self.widget._view_name is not None:
                plaintext = repr(self)
                data = {
                    'text/plain': plaintext,
                    'application/vnd.jupyter.widget-view+json': {
                        'version_major': 2,
                        'version_minor': 0,
                        'model_id': self.widget._model_id
                    }
                }
                IPython.display.display(data, raw=True)
                self.widget._handle_displayed(**kwargs)


class URLSelector(Base):
    def __init__(self, done_callback=None):
        self.done_callback = done_callback
        self.lurl = widgets.Label(value='URL:')
        self.url = widgets.Text(
            placeholder="Full URL with protocol",
            layout=widgets.Layout(flex='10 1 auto', width='auto'))
        self.x = widgets.Button(
            icon='close', tooltip='Close Selector',
            layout=widgets.Layout(flex='1 1 auto', width='auto'))
        self.x.on_click(lambda ev: self.stop())
        self.ok = widgets.Button(
            icon='check', tooltip='OK',
            layout=widgets.Layout(flex='1 1 auto', width='auto'))
        self.ok.on_click(lambda ev: self.stop(ok=self.url.value))
        self.widget = widgets.HBox(children=[self.lurl, self.url, self.ok,
                                             self.x])


class FileSelector(Base):
    """
    ipywidgets interface for picking files
    The current path is stored in ``.path`` 
    and the current selection is stored in ``.value``.
    """

    def __init__(self, done_callback=None, path=None, filters=None):
        """
        Parameters
        ----------
        done_callback : function
            Called when the tick or cross buttons are clicked. Expects
            signature func(path, ok=True|False).
        filters : list of str or None
            Only show files ending in one of these strings. Normally used for
            picking file extensions. None is an alias for [''], passes all
            files.
        """
        self.done_callback = done_callback
        if filters:
            if not isinstance(filters, (list, tuple)):
                filters = [filters]
            self.filters = list(filters)
        else:
            self.filters = ['']
        if not path or not os.path.exists(path):
            self.path = os.getcwd() + '/'
        else:
            self.path = (path + '/').replace('//', ',')
        self.main = widgets.Select(rows=7)
        self.button = widgets.Button(
            icon='chevron-left', tooltip='Parent',
            layout=widgets.Layout(flex='1 1 auto', width='auto'))
        self.button.on_click(self.up)
        self.label = widgets.Label(
            layout=widgets.Layout(flex='100 1 auto', width='auto'))
        self.x = widgets.Button(
            icon='close', tooltip='Close Selector',
            layout=widgets.Layout(width='auto'))
        self.x.on_click(lambda ev: self.stop())
        self.ok = widgets.Button(
            icon='check', tooltip='OK',
            layout=widgets.Layout(width='auto'))
        self.ok.on_click(lambda ev: self._ok())
        self.make_options()
        self.main.observe(self.changed, 'value')
        self.upper = widgets.Box(children=[self.button, self.label])
        self.right = widgets.VBox(children=[self.x, self.ok])
        self.lower = widgets.HBox(children=[self.main, self.right])
        self.widget = widgets.VBox(children=[self.upper, self.lower])
        self.ignore = False

    def _ok(self):
        fn = self.main.value
        if fn.endswith('/'):
            self.stop()
        else:
            self.stop(os.path.join(self.path, fn))

    def make_options(self):
        self.ignore = True
        self.label.value = self.path
        out = []
        for f in sorted(os.listdir(self.path)):
            if (os.path.isdir(self.path + f) and
                    not any(f.startswith(prefix) for prefix in "._~")):
                out.append(f + '/')
            elif (not any(f.startswith(prefix) for prefix in "._~") and
                  any(f.endswith(ext) for ext in self.filters + \
                                                 list(map(str.upper, self.filters)))):
                out.append(f)
        self.main.value = self.value = None
        self.fullpath = self.path
        self.main.options = out
        self.ignore = False

    def up(self, ev):
        self.path = os.path.dirname(
            self.path.rstrip('/')).rstrip('/') + '/'
        self.make_options()

    def changed(self, ev):
        if self.ignore:
            self.value = None
            self.fullpath = None
            return
        with ignore(self):
            fn = ev['new']
            if fn.endswith('/'):
                self.path = self.path + fn
                self.make_options()
            self.value = self.main.value
            self.fullpath = os.path.join(self.path, self.value)
