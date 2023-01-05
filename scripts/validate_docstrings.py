#!/usr/bin/env python3
"""
Analyze docstrings to detect errors.

If no argument is provided, it does a quick check of docstrings and returns
a csv with all API functions and results of basic checks.

If a function or method is provided in the form "spectrochempy.function",
"spectrochempy.module.class.method", etc. a list of all errors in the docstring for
the specified function or method.

Usage::
    $ scripts/validate_docstrings.py
    $ scripts/scripts/validate_docstrings.py spectrochempy.NDDataset.read

Copied and modified from https://github.com/pandas-dev/pandas/scripts/validate_docstrings.py (BSD 3-Clause License)

"""
from __future__ import annotations

import argparse
import doctest
import importlib
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile

import matplotlib
import matplotlib.pyplot as plt
import numpy
from numpydoc.docscrape import get_doc_object
from numpydoc.validate import Validator, validate

import spectrochempy

# With template backend, matplotlib plots nothing
matplotlib.use("template")

IGNORE_VALIDATION = {}
PRIVATE_CLASSES = []
ERROR_MSGS = {
    "GL04": "Private classes ({mentioned_private_classes}) should not be "
    "mentioned in public docstrings",
    "GL05": "Use 'array-like' rather than 'array_like' in docstrings.",
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


def get_api_items(api_doc_fd):
    """
    Yield information about all public API items.

    Parse api.rst file from the documentation, and extract all the functions,
    methods, classes, attributes... This should include all spectrochempy public API.

    Parameters
    ----------
    api_doc_fd : file descriptor
        A file descriptor of the API documentation page, containing the table
        of contents with all the public API.

    Yields
    ------
    name : str
        The name of the object (e.g. 'spectrochempy.NDDataset).
    func : function
        The object itself. In most cases this will be a function or method,
        but it can also be classes, properties, cython objects...
    section : str
        The name of the section in the API page where the object item is
        located.
    subsection : str
        The name of the subsection in the API page where the object item is
        located.
    """
    current_module = "spectrochempy"
    previous_line = current_section = current_subsection = ""
    position = None
    for line in api_doc_fd:
        line = line.strip()
        if len(line) == len(previous_line):
            if set(line) == set("-"):
                current_section = previous_line
                continue
            if set(line) == set("~"):
                current_subsection = previous_line
                continue

        if line.startswith(".. currentmodule::"):
            current_module = line.replace(".. currentmodule::", "").strip()
            continue

        if line == ".. autosummary::":
            position = "autosummary"
            continue

        if position == "autosummary":
            if line == "":
                position = "items"
                continue

        if position == "items":
            if line == "":
                position = None
                continue
            item = line.strip()
            if item in IGNORE_VALIDATION:
                continue
            func = importlib.import_module(current_module)
            for part in item.split("."):
                func = getattr(func, part)

            yield (
                ".".join([current_module, item]),
                func,
                current_section,
                current_subsection,
            )

        previous_line = line


class spectrochempyDocstring(Validator):
    def __init__(self, func_name: str, doc_obj=None) -> None:
        self.func_name = func_name
        if doc_obj is None:
            doc_obj = get_doc_object(Validator._load_obj(func_name))
        super().__init__(doc_obj)

    @property
    def name(self):
        return self.func_name

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
            print(
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
        # that do not user numpy or spectrochempy
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
            response = subprocess.run(cmd, capture_output=True, check=False, text=True)
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


def spectrochempy_validate(func_name: str):
    """
    Call the numpydoc validation, and add the errors specific to spectrochempy.

    Parameters
    ----------
    func_name : str
        Name of the object of the docstring to validate.

    Returns
    -------
    dict
        Information about the docstring and the errors found.
    """
    func_obj = Validator._load_obj(func_name)
    # Some objects are instances, e.g. IndexSlice, which numpydoc can't validate
    doc_obj = get_doc_object(func_obj, doc=func_obj.__doc__)
    doc = spectrochempyDocstring(func_name, doc_obj)
    result = validate(doc_obj)

    mentioned_errs = doc.mentioned_private_classes
    if mentioned_errs:
        result["errors"].append(
            spectrochempy_error(
                "GL04", mentioned_private_classes=", ".join(mentioned_errs)
            )
        )

    if doc.see_also:
        for rel_name in doc.see_also:
            if rel_name.startswith("spectrochempy."):
                result["errors"].append(
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
            result["errors"].append(
                spectrochempy_error("EX02", doctest_log=result["examples_errs"])
            )

        for error_code, error_message, error_count in doc.validate_pep8():
            times_happening = f" ({error_count} times)" if error_count > 1 else ""
            result["errors"].append(
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
                result["errors"].append(
                    spectrochempy_error("EX04", imported_library=wrong_import)
                )

    if doc.non_hyphenated_array_like():
        result["errors"].append(spectrochempy_error("GL05"))

    plt.close("all")
    return result


def validate_all(prefix, ignore_deprecated=False):
    """
    Execute the validation of all docstrings, and return a dict with the
    results.

    Parameters
    ----------
    prefix : str or None
        If provided, only the docstrings that start with this pattern will be
        validated. If None, all docstrings will be validated.
    ignore_deprecated: bool, default False
        If True, deprecated objects are ignored when validating docstrings.

    Returns
    -------
    dict
        A dictionary with an item for every function/method... containing
        all the validation information.
    """
    result = {}
    seen = {}

    base_path = pathlib.Path(__file__).parent.parent
    api_doc_fnames = pathlib.Path(base_path, "docs", "userguide", "reference")
    api_items = []
    for api_doc_fname in api_doc_fnames.glob("*.rst"):
        with open(api_doc_fname) as f:
            api_items += list(get_api_items(f))

    for func_name, _, section, subsection in api_items:
        if prefix and not func_name.startswith(prefix):
            continue
        doc_info = spectrochempy_validate(func_name)
        if ignore_deprecated and doc_info["deprecated"]:
            continue
        result[func_name] = doc_info

        shared_code_key = doc_info["file"], doc_info["file_line"]
        shared_code = seen.get(shared_code_key, "")
        result[func_name].update(
            {
                "in_api": True,
                "section": section,
                "subsection": subsection,
                "shared_code_with": shared_code,
            }
        )

        seen[shared_code_key] = func_name

    return result


def print_validate_all_results(
    prefix: str,
    errors: list[str] | None,
    output_format: str,
    ignore_deprecated: bool,
):
    if output_format not in ("default", "json", "actions"):
        raise ValueError(f'Unknown output_format "{output_format}"')

    result = validate_all(prefix, ignore_deprecated)

    if output_format == "json":
        sys.stdout.write(json.dumps(result))
        return 0

    prefix = "##[error]" if output_format == "actions" else ""
    exit_status = 0
    for name, res in result.items():
        for err_code, err_desc in res["errors"]:
            if errors and err_code not in errors:
                continue
            sys.stdout.write(
                f'{prefix}{res["file"]}:{res["file_line"]}:'
                f"{err_code}:{name}:{err_desc}\n"
            )
            exit_status += 1

    return exit_status


def print_validate_one_results(func_name: str):
    def header(title, width=80, char="#"):
        full_line = char * width
        side_len = (width - len(title) - 2) // 2
        adj = "" if len(title) % 2 == 0 else " "
        title_line = f"{char * side_len} {title}{adj} {char * side_len}"

        return f"\n{full_line}\n{title_line}\n{full_line}\n\n"

    result = spectrochempy_validate(func_name)

    sys.stderr.write(header(f"Docstring ({func_name})"))
    sys.stderr.write(f"{result['docstring']}\n")

    sys.stderr.write(header("Validation"))
    if result["errors"]:
        sys.stderr.write(f'{len(result["errors"])} Errors found:\n')
        for err_code, err_desc in result["errors"]:
            if err_code == "EX02":  # Failing examples are printed at the end
                sys.stderr.write("\tExamples do not pass tests\n")
                continue
            sys.stderr.write(f"\t{err_desc}\n")
    else:
        sys.stderr.write(f'Docstring for "{func_name}" correct. :)\n')

    if result["examples_errs"]:
        sys.stderr.write(header("Doctests"))
        sys.stderr.write(result["examples_errs"])


def main(func_name, prefix, errors, output_format, ignore_deprecated):
    """
    Main entry point. Call the validation for one or for all docstrings.
    """
    if func_name is None:
        return print_validate_all_results(
            prefix, errors, output_format, ignore_deprecated
        )
    else:
        print_validate_one_results(func_name)
        return 0


if __name__ == "__main__":
    format_opts = "default", "json", "actions"
    func_help = (
        "function or method to validate (e.g. spectrochempy.DataFrame.head) "
        "if not provided, all docstrings are validated and returned "
        "as JSON"
    )
    argparser = argparse.ArgumentParser(description="validate spectrochempy docstrings")
    argparser.add_argument("function", nargs="?", default=None, help=func_help)
    argparser.add_argument(
        "--format",
        default="default",
        choices=format_opts,
        help="format of the output when validating "
        "multiple docstrings (ignored when validating one). "
        "It can be {str(format_opts)[1:-1]}",
    )
    argparser.add_argument(
        "--prefix",
        default=None,
        help="pattern for the "
        "docstring names, in order to decide which ones "
        'will be validated. A prefix "spectrochempy.Series.str."'
        "will make the script validate all the docstrings "
        "of methods starting by this pattern. It is "
        "ignored if parameter function is provided",
    )
    argparser.add_argument(
        "--errors",
        default=None,
        help="comma separated "
        "list of error codes to validate. By default it "
        "validates all errors (ignored when validating "
        "a single docstring)",
    )
    argparser.add_argument(
        "--ignore_deprecated",
        default=False,
        action="store_true",
        help="if this flag is set, "
        "deprecated objects are ignored when validating "
        "all docstrings",
    )

    args = argparser.parse_args()
    sys.exit(
        main(
            args.function,
            args.prefix,
            args.errors.split(",") if args.errors else None,
            args.format,
            args.ignore_deprecated,
        )
    )
