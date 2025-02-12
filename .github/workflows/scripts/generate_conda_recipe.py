import textwrap
from pathlib import Path

import toml
from jinja2 import Template
from setuptools_scm import get_version

# Define paths
repo_path = Path(__file__).parent.parent.parent.parent
template_dir = repo_path / ".ci" / "templates"

# Load template
meta_template_file = template_dir / "meta.tmpl"
meta_template = Template(
    meta_template_file.read_text("utf-8"),
    keep_trailing_newline=True,
)

# Load pyproject.toml
pyproject_file = repo_path / "pyproject.toml"
pyproject = toml.load(pyproject_file)

# Get dependencies from pyproject.toml
deps = pyproject["project"]["dependencies"]

# Mapping between pip and conda package names
renaming = {  # pip to conda
    "gitpython": "git",  # Full package name
    "numpy-quaternion": "quaternion",  # Full package name
    "matplotlib": "matplotlib-base",  # Base package name
}


# Function to process dependency string
def process_dependency(dep):
    # Split package name from version specification
    parts = dep.split(">=")
    if len(parts) == 1:
        pkg_name = dep
        version_spec = ""
    else:
        pkg_name = parts[0].strip()
        version_spec = f">={parts[1]}"

    # Apply renaming if package is in the mapping
    conda_name = renaming.get(pkg_name, pkg_name)

    # Return full dependency string
    return f"{conda_name} {version_spec}".strip()


# Process dependencies
deps = [process_dependency(dep) for dep in deps]
deps_strg = "    - " + "\n    - ".join(deps)

# Get version string from setuptools_scm
pvs = get_version()
# print(f"Current version string = {pvs}")
latest = pvs.split("+")[0]
version_parts = latest.split(".")
version = f"{version_parts[0]}.{version_parts[1]}.{version_parts[2]}"
devstring = version_parts[3] if len(version_parts) > 3 else "stable"

# Get README content
readme_file = repo_path / "README.md"
readme = readme_file.read_text()
readme_strg = textwrap.indent(readme, "    ")

# Update meta.yaml using jinja2 template
meta_content = meta_template.render(
    version=version,
    devstring=devstring,
    dependencies=deps_strg,
    pyproject=pyproject,
    readme=readme_strg,
)

# Write updated meta.yaml to file
meta_filename = repo_path / "recipe" / "meta.yaml"
meta_filename.write_text(meta_content)
# print(f"Updated {meta_filename}")
