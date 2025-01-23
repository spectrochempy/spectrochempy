import os

from setuptools_scm import get_version

# Get version string from setuptools_scm
pvs = get_version()
print(f"Current version string = {pvs}")

# Extract components
latest = pvs.split("+")[0]
version_parts = latest.split(".")
version = f"{version_parts[0]}.{version_parts[1]}.{version_parts[2]}"

devstring = version_parts[3] if len(version_parts) > 3 else "stable"

print(latest)
print(version)
print(devstring)

try:
    # Export variable in the GitHub environment
    with open(os.getenv("GITHUB_ENV"), "a") as env_file:
        env_file.write(f"VERSIONSTRING={latest}\n")
        env_file.write(f"VERSION={version}\n")
        env_file.write(f"DEVSTRING={devstring}\n")
except TypeError as e:
    print(f"Error: {e}")
    print("This script must be run in a GitHub Actions workflow.")
    exit(1)
