# This script is used to extract the version string from the setuptools_scm
# generated version and export it as an environment variable in the GitHub
# Actions workflow.
# ruff: noqa: T201

import os

from setuptools_scm import get_version

# Get version string from setuptools_scm
pvs = get_version()
print(f"Current version string = {pvs}")

# Extract components
latest = pvs.split("+")[0]
version_parts = latest.split(".")
version = f"{version_parts[0]}.{version_parts[1]}.{version_parts[2]}"

# Determine if the version is a development version or stable
devstring = version_parts[3] if len(version_parts) > 3 else "stable"

print(latest)
print(version)
print(devstring)

try:
    # Export variables in the GitHub environment
    with open(os.getenv("GITHUB_ENV"), "a") as env_file:
        env_file.write(f"VERSIONSTRING={latest}\n")
        env_file.write(f"VERSION={version}\n")
        env_file.write(f"DEVSTRING={devstring}\n")
except TypeError as e:
    # Handle error if the script is not run in a GitHub Actions workflow
    print(f"Error: {e}")
    print("This script must be run in a GitHub Actions workflow.")
    exit(1)
