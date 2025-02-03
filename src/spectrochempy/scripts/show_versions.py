#!/usr/bin/env python
import sys

from spectrochempy.utils.print_versions import show_versions


def main():
    """Run the main entry point for the script."""
    show_versions(file=sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
