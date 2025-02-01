# ruff: noqa: T201,S603
import argparse
import sys
from subprocess import run


def main():
    """Build documentation for SpectroChemPy."""
    parser = argparse.ArgumentParser(
        description="Build documentation for SpectroChemPy"
    )

    parser.add_argument(
        "command", nargs="?", default="html", help="available commands: html, clean"
    )

    args = parser.parse_args()

    if args.command == "html":
        build_html()
    elif args.command == "clean":
        clean()
    else:
        parser.print_help(sys.stderr)
        print(f"Unknown command {args.command}.")
        return 1

    return 0


def build_html():
    """Build HTML documentation using Sphinx."""
    import shlex

    run(
        [
            shlex.quote(arg)
            for arg in ["sphinx-build", "-j1", "-b", "html", "docs", "docs/_build/html"]
        ],
        check=True,
    )


def clean():
    """Clean/remove the built documentation."""
    import shutil
    from pathlib import Path

    DOCS = Path(__file__).parent
    HTML = DOCS / "_build" / "html"
    DOCTREES = DOCS / "_build" / "doctrees"

    shutil.rmtree(HTML, ignore_errors=True)
    print(f"removed {HTML}")
    shutil.rmtree(DOCTREES, ignore_errors=True)
    print(f"removed {DOCTREES}")


if __name__ == "__main__":
    sys.exit(main())
