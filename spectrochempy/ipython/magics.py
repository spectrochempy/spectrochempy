# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Magic ipython Classes
"""
from IPython.core.error import UsageError
from IPython.core.magic import Magics, line_cell_magic, magics_class
from IPython.core.magics.code import extract_symbols
from IPython.utils.text import get_text_list

from spectrochempy import error_, warning_

__all__ = ["SpectroChemPyMagics"]


# ======================================================================================
# Magic ipython Classes
# ======================================================================================
@magics_class
class SpectroChemPyMagics(Magics):
    """
    This class implements the addscript ipython magic function.

    The ipython extensions`can be loaded via `%load_ext spectrochempy.ipython`
    or be configured to be autoloaded by IPython at startup time.
    """

    @line_cell_magic
    def addscript(self, pars="", cell=None):
        """
        This works both as **%addscript** and as **%%addscript**.

        This magic command can either take a local filename, element in the
        namespace or history range (see %history),
        or the current cell content.


        Usage:

            %addscript  -p project  n1-n2 n3-n4 ... n5 .. n6 ...

            or

            %%addscript -p project
            ...code lines ...


        Options:

            -p <string>         Name of the project where the script will be stored.
                                If not provided, a project with a standard
                                name : `proj` is searched.
            -o <string>         script name.
            -s <symbols>        Specify function or classes to load from python
                                source.
            -a                  append to the current script instead of
                                overwriting it.
            -n                  Search symbol in the current namespace.


        Examples
        --------

        .. sourcecode:: ipython

           In[1]: %addscript myscript.py

           In[2]: %addscript 7-27

           In[3]: %addscript -s MyClass,myfunction myscript.py
           In[4]: %addscript MyClass

           In[5]: %addscript mymodule.myfunction
        """
        opts, args = self.parse_options(pars, "p:o:s:n:a")

        # append = 'a' in opts
        # mode = 'a' if append else 'w'
        search_ns = "n" in opts

        if not args and not cell and not search_ns:  # pragma: no cover
            raise UsageError(
                "Missing filename, input history range, "
                "or element in the user namespace.\n "
                "If no argument are given then the cell content "
                "should "
                "not be empty"
            )
        name = "script"
        if "o" in opts:
            name = opts["o"]

        proj = "proj"
        if "p" in opts:
            proj = opts["p"]
        if proj not in self.shell.user_ns:  # pragma: no cover
            raise ValueError(
                f"Cannot find any project with name `{proj}` in the namespace."
            )
        # get the proj object
        projobj = self.shell.user_ns[proj]

        contents = ""
        if search_ns:
            contents += (
                "\n" + self.shell.find_user_code(opts["n"], search_ns=search_ns) + "\n"
            )

        args = " ".join(args)
        if args.strip():
            contents += (
                "\n" + self.shell.find_user_code(args, search_ns=search_ns) + "\n"
            )

        if "s" in opts:  # pragma: no cover
            try:
                blocks, not_found = extract_symbols(contents, opts["s"])
            except SyntaxError:
                # non python code
                error_(SyntaxError, "Unable to parse the input as valid Python code")
                return None

            if len(not_found) == 1:
                warning_(f"The symbol `{not_found[0]}` was not found")
            elif len(not_found) > 1:
                sym = get_text_list(not_found, wrap_item_with="`")
                warning_(f"The symbols {sym} were not found")

            contents = "\n".join(blocks)

        if cell:
            contents += "\n" + cell

        # import delayed to avoid circular import error
        from spectrochempy.core.script import Script

        script = Script(name, content=contents)
        projobj[name] = script

        return f"Script {name} created."

        # @line_magic  # def runscript(self, pars=''):  #     """  #  #  # """
        #     opts,
        # args = self.parse_options(pars, '')  #  #     if  # not args:
        #         raise UsageError('Missing script
        # name')  #  #  # return args


def load_ipython_extension(ipython):
    """
    The ipython extensions`can be loaded via `%load_ext spectrochempy.ipython`
    or be configured to be autoloaded by IPython at startup time.
    """
    ipython.register_magics(SpectroChemPyMagics)
