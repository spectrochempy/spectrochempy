import re
import textwrap

import docrep

common_doc = """
copy : bool, optional, Default: False
    Perform a copy of the passed object.
inplace : bool, optional, default: False
    By default, the method returns a newly allocated object.
    If `inplace` is set to True, the input object is returned.
**kwargs : keyword parameters, optional
    See Other Parameters.
"""


class DocstringProcessor(docrep.DocstringProcessor):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        regex = re.compile(r"(?=^[*]{0,2}\b\w+\b\s?:?\s?)", re.MULTILINE | re.DOTALL)
        plist = regex.split(common_doc.strip())[1:]
        params = {
            k.strip("*"): f"{k.strip()} : {v.strip()}"
            for k, v in (re.split(r"\s?:\s?", p, maxsplit=1) for p in plist)
        }
        self.params.update(params)
        self.params.update(
            {
                "out": "object\n"
                "    Input object or a newly allocated object\n"
                "    depending on the `inplace` flag.",
                "new": "object\n" "    Newly allocated object.",
            }
        )

    def dedent(self, s, stacklevel=3):
        s_ = s
        start = ""
        end = ""
        string = True
        if not isinstance(s, str) and hasattr(s, "__doc__"):
            string = False
            s_ = s.__doc__
        if s_.startswith("\n"):  # restore the first blank line
            start = "\n"
        if s_.strip(" ").endswith("\n"):  # restore the last return before quote
            end = "\n"
        s_mod = super().dedent(s, stacklevel=stacklevel)
        if string:
            s_mod = f"{start}{s_mod}{end}"
        else:
            s_mod.__doc__ = f"{start}{s_mod.__doc__}{end}"
        return s_mod


# Docstring substitution (docrep)
# --------------------------------------------------------------------------------------
_docstring = DocstringProcessor()


# TODO replace this in module where it is used by docrep
def add_docstring(*args):
    """
    Decorator which add a docstring to the actual func doctring.
    """

    def new_doc(func):

        for item in args:
            item.strip()

        func.__doc__ = textwrap.dedent(func.__doc__).format(*args)
        return func

    return new_doc
