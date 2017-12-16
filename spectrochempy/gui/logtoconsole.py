# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================



import sys
import logging
from ..application import app
log = app.log
from ..extern.pyqtgraph.Qt import QtCore


class QtHandler(logging.Handler):
    def __init__(self):
        logging.Handler.__init__(self)

    def emit(self, record):
        record = self.format(record)
        if record:
            msg = '%s' % record
            color = '#000088'
            if 'ERROR |' in msg:
                color = '#EE0000'
            elif 'WARNING |' in msg:
                color = '#880000'
            msg = "<font color={}>{}</font><br>".format(color, msg)

            ConsoleStream.stdout().write(msg)



class ConsoleStream(QtCore.QObject):
    _stdout = None
    _stderr = None

    messageWritten = QtCore.pyqtSignal(str, bool)

    def flush(self):
        pass

    def fileno(self):
        return -1

    def write(self, msg, html=True):
        if (not self.signalsBlocked()):
            self.messageWritten.emit(msg, html)

    @staticmethod
    def stdout():
        if (not ConsoleStream._stdout):
            ConsoleStream._stdout = ConsoleStream()
            sys.stdout = ConsoleStream._stdout
        return ConsoleStream._stdout

    @staticmethod
    def stderr():
        if (not ConsoleStream._stderr):
            ConsoleStream._stderr = ConsoleStream()
            sys.stderr = ConsoleStream._stderr
        return ConsoleStream._stderr

def redirectoutput(console=None):
    # stout, sterr redirect
    if console is None:
        log.error("A console is needed for redirecting output")
        return

    ConsoleStream.stdout().messageWritten.connect(console.write)
    ConsoleStream.stderr().messageWritten.connect(console.write)

