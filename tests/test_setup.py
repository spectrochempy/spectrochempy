# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT  
# See full LICENSE agreement in the root directory
# =============================================================================


# run setup in develop mode

import contextlib
import sys
import pytest

NO = True  # TODO: it break the installation of SpectroChemPy

do_it = not NO and ('-c' not in sys.argv[0] ) # pytest not in parallel mode

@pytest.mark.skipif(not do_it, reason="ignore this during all tests in parallel")
def test_setup():

    @contextlib.contextmanager
    def redirect_argv(new):
        sys._argv = sys.argv[:]
        sys.argv.append(str(new))
        sys.argv = sys.argv[1:]
        yield
        sys.argv = sys._argv

    with redirect_argv('develop'):
        print((sys.argv))
        import __setup__ as s
        s.run_setup()