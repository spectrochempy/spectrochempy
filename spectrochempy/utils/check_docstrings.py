"""
Analyze docstrings to detect errors.

Adapted from Pandas (see License in the root directory)
"""
import doctest
import io
import inspect
import os
import pathlib
import subprocess
import tempfile

import traceback

import matplotlib
import matplotlib.pyplot as plt
import numpy
from numpydoc.docscrape import get_doc_object
from numpydoc.validate import Validator, validate, error

import spectrochempy

# With template backend, matplotlib plots nothing
matplotlib.use("template")


PRIVATE_CLASSES = [
    "HasTraits",
]
ERROR_MSGS = {
    "GL04": "Private classes ({mentioned_private_classes}) should not be "
    "mentioned in public docstrings",
    "GL05": "Use 'array-like' rather than 'array_like' in docstrings.",
    "GL11": "Other Parameters section missing while `**kwargs` is in "
    "class or method signature.",
    "SA05": "{reference_name} in `See Also` section does not need `spectrochempy` "
    "prefix, use {right_reference} instead.",
    "EX02": "Examples do not pass tests:\n{doctest_log}",
    "EX03": "flake8 error: {error_code} {error_message}{times_happening}",
    "EX04": "Do not import {imported_library}, as it is imported "
    "automatically for the examples (numpy as np, spectrochempy as scp)",
}


def spectrochempy_error(code, **kwargs):
    """
    Copy of the numpydoc error function, since ERROR_MSGS can't be updated
    with our custom errors yet.
    """
    return (code, ERROR_MSGS[code].format(**kwargs))


class SpectroChemPyDocstring(Validator):
    def __init__(self, func_name, doc_obj=None):
        self.func_name = func_name
        if doc_obj is None:
            doc_obj = get_doc_object(Validator._load_obj(func_name))
        super().__init__(doc_obj)

    @property
    def name(self):
        return self.func_name

    @property
    def has_kwargs(self):
        return "**kwargs" in self.signature_parameters

    @property
    def mentioned_private_classes(self):
        return [klass for klass in PRIVATE_CLASSES if klass in self.raw_doc]

    @property
    def examples_errors(self):
        flags = doctest.NORMALIZE_WHITESPACE | doctest.IGNORE_EXCEPTION_DETAIL
        finder = doctest.DocTestFinder()
        runner = doctest.DocTestRunner(optionflags=flags)
        context = {"np": numpy, "scp": spectrochempy}
        error_msgs = ""
        current_dir = set(os.listdir())
        for test in finder.find(self.raw_doc, self.name, globs=context):
            f = io.StringIO()
            runner.run(test, out=f.write)
            error_msgs += f.getvalue()
        leftovers = set(os.listdir()).difference(current_dir)
        if leftovers:
            for leftover in leftovers:
                path = pathlib.Path(leftover).resolve()
                if path.is_dir():
                    path.rmdir()
                elif path.is_file():
                    path.unlink(missing_ok=True)
            raise Exception(
                f"The following files were leftover from the doctest: "
                f"{leftovers}. Please use # doctest: +SKIP"
            )
        return error_msgs

    @property
    def examples_source_code(self):
        lines = doctest.DocTestParser().get_examples(self.raw_doc)
        return [line.source for line in lines]

    def validate_pep8(self):
        if not self.examples:
            return

        # F401 is needed to not generate flake8 errors in examples
        # that do not use numpy or spectrochempy
        content = "".join(
            (
                "import numpy as np  # noqa: F401\n",
                "import spectrochempy as scp  # noqa: F401\n",
                *self.examples_source_code,
            )
        )

        error_messages = []
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as file:
            file.write(content)
            file.flush()
            cmd = ["python", "-m", "flake8", "--quiet", "--statistics", file.name]
            response = subprocess.run(cmd, capture_output=True, text=True)
            stdout = response.stdout
            stdout = stdout.replace(file.name, "")
            messages = stdout.strip("\n")
            if messages and messages != "0":
                error_messages.append(messages)

        for error_message in error_messages:
            error_count, error_code, message = error_message.split(maxsplit=2)
            yield error_code, message, int(error_count)

    def non_hyphenated_array_like(self):
        return "array_like" in self.raw_doc


def remove_errors(errs, errors=[]):
    dic_errs = dict(errs)
    if not isinstance(errors, list):
        errors = [errors]
    for err in errors:
        dic_errs.pop(err, None)
    errs = list(dic_errs.items())
    return errs


def spectrochempy_validate(func_name, exclude=[]):
    """
    Call the numpydoc validation, and add the errors specific to spectrochempy.

    Parameters
    ----------
    func_name : str
        Name of the object of the docstring to validate.
    exclude : list
        List of error code to exclude, e.g. ["SA01", ...].

    Returns
    -------
    dict
        Information about the docstring and the errors found.
    """
    func_obj = Validator._load_obj(func_name)
    doc_obj = get_doc_object(func_obj)
    doc = SpectroChemPyDocstring(func_name, doc_obj)
    result = validate(doc_obj)
    errs = result["errors"]

    mentioned_errs = doc.mentioned_private_classes
    if mentioned_errs:
        errs.append(
            spectrochempy_error(
                "GL04", mentioned_private_classes=", ".join(mentioned_errs)
            )
        )

    has_kwargs = doc.has_kwargs
    if has_kwargs:
        errs = remove_errors(errs, "PR02")
        if not doc.doc_other_parameters:
            errs.append(spectrochempy_error("GL11"))

    if exclude:
        errs = remove_errors(errs, exclude)

    if doc.see_also:
        for rel_name in doc.see_also:
            if rel_name.startswith("spectrochempy."):
                errs.append(
                    spectrochempy_error(
                        "SA05",
                        reference_name=rel_name,
                        right_reference=rel_name[len("spectrochempy.") :],
                    )
                )

    result["examples_errs"] = ""
    if doc.examples:
        result["examples_errs"] = doc.examples_errors
        if result["examples_errs"]:
            errs.append(
                spectrochempy_error("EX02", doctest_log=result["examples_errs"])
            )

        for error_code, error_message, error_count in doc.validate_pep8():
            times_happening = f" ({error_count} times)" if error_count > 1 else ""
            errs.append(
                spectrochempy_error(
                    "EX03",
                    error_code=error_code,
                    error_message=error_message,
                    times_happening=times_happening,
                )
            )
        examples_source_code = "".join(doc.examples_source_code)
        for wrong_import in ("numpy", "spectrochempy"):
            if f"import {wrong_import}" in examples_source_code:
                errs.append(spectrochempy_error("EX04", imported_library=wrong_import))

    if doc.non_hyphenated_array_like():
        errs.append(spectrochempy_error("GL05"))

    # cases where docrep dedent was used
    if error("GL01") in errs and not doc.raw_doc.startswith(""):
        errs = remove_errors(errs, "GL01")
    if error("GL02") in errs and not doc.raw_doc.startswith(""):
        errs = remove_errors(errs, "GL02")

    # case of properties (we accept a single line summary)
    if hasattr(doc.code_obj, "fget"):
        errs = remove_errors(errs, "ES01")

    result["errors"] = errs
    plt.close("all")
    if result["file"] is None:
        # sometimes it is because the code_obj is a property
        if hasattr(doc.code_obj, "fget"):
            try:
                result["file"] = inspect.getsourcefile(doc.code_obj.fget)
                result["file_line"] = inspect.getsourcelines(doc.code_obj.fget)[-1]
            except (OSError, TypeError):
                pass

    return result


def check_docstrings(module, obj, exclude=[]):
    members = [f"{module}.{obj.__name__}"]
    print(module)
    print(obj.__name__)
    for m in dir(obj):
        member = getattr(obj, m)
        if not m.startswith("_") and (
            (
                member.__class__.__name__ == "property"
                or (hasattr(member, "__module__") and member.__module__ == module)
            )
            and m not in ["cross_validation_lock"]
        ):
            members.append(f"{module}.{obj.__name__}.{m}")
            print(f"{obj.__name__}.{m}")

    for member in members:
        result = spectrochempy_validate(member, exclude=exclude)
        if result["errors"]:
            result["member_name"] = member
            raise DocstringError(result)


class DocstringError(Exception):
    def __init__(self, result):

        message = ""
        message += f"{len(result['errors'])} DocstringError(s) found:\n"
        message += f"{' '*10}{'-'*26}\n"
        for err_code, err_desc in result["errors"]:
            if err_code == "EX02":  # Failing examples are printed at the end
                message += f"{' '*2}Examples do not pass tests\n"
                continue
            message += f"{' '*10}* {err_code}: {err_desc}\n"
        if result["examples_errs"]:
            message += "\n\nDoctests:\n---------\n"
            message += result["examples_errs"]

        traceback_details = {
            "filename": result["file"],
            "lineno": result["file_line"],
            "name": result["member_name"],
            "type": "DocstringError",
            "message": message,
        }

        traceback.format_exc()  # cannot be used with pytest in debug mode

        traceback_template = """
        Docstring format error:
          File "%(filename)s", line %(lineno)s,
          in %(name)s.
          %(message)s\n
        """
        print(traceback_template % traceback_details)
