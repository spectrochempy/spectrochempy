# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

from spectrochempy.utils import docstrings

def test_docstrings():

    @docstrings.get_sectionsf('do_something')
    @docstrings.dedent
    def do_something(a, b, c, d):
        """
        Add tree numbers

        Parameters
        ----------
        a : int
            The first number
        b : int
            The second number
        c : int
            The third number
        d : int
            Another number

        Returns
        -------
        int
            `a` + `b` + `c` + `d`

        """
        return a + b + c + d


    docstrings.delete_params('do_something.parameters', 'c', 'd')


    @docstrings.dedent
    def do_more(*args, **kwargs):
        """
        Add two numbers and multiply it by 2

        Parameters
        ----------
        %(do_something.parameters.no_c|d)s

        Returns
        -------
        int
            (`a` + `b`) * 2

        """

        return do_something(*args[:1]) * 2


    print(do_more.__doc__)







# ======================================================================================================================
if __name__ == '__main__':
    pass
