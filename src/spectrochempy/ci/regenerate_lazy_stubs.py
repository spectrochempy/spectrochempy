# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: T201

import argparse
import os
from pathlib import Path

excluded_packages = ["examples", "ci", "data", "lazyimport"]


def extract_copyright_header(file_path):
    """Extract copyright header from an existing file."""
    if not os.path.exists(file_path):
        return ""

    with open(file_path) as f:
        content = f.read()

    # Try to match a standard copyright header (lines starting with #)
    header_lines = []
    in_header = False

    for line in content.splitlines():
        if line.startswith("# =====") and not in_header:
            in_header = True
            header_lines.append(line)
        elif in_header and line.startswith("#"):
            header_lines.append(line)
        elif in_header:
            break

    return "\n".join(header_lines) + "\n" if header_lines else ""


def check_and_update_init_py(package_path):
    """
    Check if __init__.py has lazy loading setup, and update it if not.

    Returns True if the file was updated, False if it was already correct.
    """
    init_py_path = package_path / "__init__.py"
    if not init_py_path.exists():
        return False

    with open(init_py_path) as f:
        content = f.read()

    # Check if lazy_loader is already imported and attach_stub is used
    if "import lazy_loader" in content and "attach_stub" in content:
        return False  # Already using lazy loading, no update needed

    # Extract existing copyright header
    header = extract_copyright_header(init_py_path)
    if not header:
        header = """# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""

    # Create the new content with lazy loading
    new_content = [
        header.rstrip(),
        "",
        "import lazy_loader as _lazy_loader",
        "",
        "# --------------------------------------------------------------------------------------",
        "# Lazy loading of sub-packages",
        "# --------------------------------------------------------------------------------------",
        "__getattr__, __dir__, __all__ = _lazy_loader.attach_stub(__name__, __file__)",
        "",
    ]

    # Write the file
    with open(init_py_path, "w") as f:
        f.write("\n".join(new_content))

    print(f"Updated {init_py_path} to use lazy loading")
    return True


def should_skip_package(package_path):
    """Check if a package should be skipped for lazy loading."""
    package_path = Path(package_path).resolve()

    # Skip if the package name is in the excluded list
    if package_path.name in excluded_packages:
        print(f"Skipping excluded package: {package_path.name}")
        return True

    init_py_path = package_path / "__init__.py"

    # Check if there's a skip marker in __init__.py
    if init_py_path.exists():
        with open(init_py_path) as f:
            content = f.read()
            if "# lazy_stub: skip" in content:
                print(f"Skipping package with skip marker: {package_path}")
                return True

    # Check if the package only contains __init__.py (empty package)
    py_files = list(package_path.glob("*.py"))
    subpackages = [
        d for d in package_path.iterdir() if d.is_dir() and (d / "__init__.py").exists()
    ]

    if len(py_files) == 1 and py_files[0].name == "__init__.py" and not subpackages:
        print(f"Skipping empty package (only __init__.py): {package_path}")
        return True

    return False


def generate_init_pyi(package_path):
    """Generate __init__.pyi for lazy loading for the given package path."""
    package_path = Path(package_path).resolve()

    # Check if the directory exists and is a package
    if not package_path.is_dir():
        print(f"Error: {package_path} is not a directory")
        return False

    init_py_path = package_path / "__init__.py"
    if not init_py_path.exists():
        print(f"Error: {package_path} is not a Python package (no __init__.py found)")
        return False

    # Check if we should skip this package
    if should_skip_package(package_path):
        return False

    # Check and update __init__.py if needed
    check_and_update_init_py(package_path)

    # Find all potential modules (Python files) and subpackages in the directory
    modules = []
    for item in package_path.iterdir():
        # Skip special files
        if item.name == "__init__.py" or item.name.startswith("__"):
            continue

        if item.name in excluded_packages:
            continue

        # Check if it's a Python module
        if item.suffix == ".py":
            module_name = item.stem
            modules.append(module_name)
        # Check if it's a subpackage (directory with __init__.py)
        elif item.is_dir() and (item / "__init__.py").exists():
            modules.append(item.name)

    # If no modules were found, don't generate an empty stub
    if not modules:
        print(f"Skipping package with no modules/subpackages: {package_path}")
        return False

    # Sort modules for consistency
    modules.sort()

    # Path for the output file
    init_pyi_path = package_path / "__init__.pyi"

    # Extract existing header if the file exists
    header = extract_copyright_header(init_pyi_path)
    if not header:
        # If no existing header, try to use the one from __init__.py
        header = extract_copyright_header(init_py_path)

    # If still no header, use the standard copyright header
    if not header:
        header = """# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""

    # Generate the content
    content = [
        header.rstrip(),  # Ensure no trailing spaces
        "",  # Add a blank line after the header
        "# ruff: noqa\n",
        "__all__ = [",
    ]

    # Add modules to __all__
    for module in modules:
        content.append(f'    "{module}",')

    content.append("]\n")

    # Add import statements
    for module in modules:
        content.append(f"from . import {module}")

    # Write the file
    with open(init_pyi_path, "w") as f:
        f.write("\n".join(content))
        f.write("\n")  # add ending new line

    print(f"Generated {init_pyi_path} with {len(modules)} modules/packages")
    return True


def find_subpackages(package_path):
    """Find all subpackages within the given package path."""
    subpackages = []
    package_path = Path(package_path).resolve()

    for item in package_path.iterdir():
        if item.is_dir() and (item / "__init__.py").exists():
            if item.name in excluded_packages:
                continue

            # Skip packages that should be excluded based on our criteria
            if not should_skip_package(item):
                subpackages.append(item)

    return subpackages


def process_package_recursive(package_path, depth=0):
    """Process a package and all its subpackages recursively."""
    success = generate_init_pyi(package_path)
    indent = "  " * depth

    if not success:
        return 0

    count = 1  # Count this package

    # Find all subpackages
    subpackages = find_subpackages(package_path)

    # Process each subpackage
    for subpackage in subpackages:
        print(f"{indent}Processing subpackage: {subpackage.name}")
        count += process_package_recursive(subpackage, depth + 1)

    return count


def main():
    """Run the script to generate lazy loading stubs."""
    parser = argparse.ArgumentParser(
        description="Generate lazy loading stubs for Python packages"
    )
    parser.add_argument("packages", nargs="*", help="Paths to packages to process")
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Process packages recursively (including all subpackages)",
    )

    args = parser.parse_args()

    package_paths = args.packages
    if not package_paths:
        # If no arguments, use the current working directory
        print("No package paths specified. Using current directory.")
        package_paths = [os.getcwd()]

    total_count = 0
    for path in package_paths:
        print(f"Processing package: {path}")
        if args.recursive:
            count = process_package_recursive(path)
            print(f"Processed {count} packages/subpackages in {path}")
            total_count += count
        else:
            if generate_init_pyi(path):
                print(f"Successfully processed {path}")
                total_count += 1
            else:
                print(f"Failed to process {path}")

    print(f"Total packages processed: {total_count}")


if __name__ == "__main__":
    import sys

    sys.argv = [__file__, "src/spectrochempy", "-r"]
    main()
