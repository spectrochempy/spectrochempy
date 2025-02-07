# Script to regenerate docs for older versions according to changes made in the documentation site structure
# ruff: noqa: T201,S603

import subprocess
import tempfile
from pathlib import Path

# Constants
PROJECT = Path(__file__).parent.parent.parent
TEMPDIRS = PROJECT / ".tempdirs"
DOCS = PROJECT / "docs"
SRC = PROJECT / "src"
# DOCS_STRUCTURE = {
# "docs": ["_static", "_templates", "sphinxext", "apigen.py", "make.py", "conf.py", "readme.md"],
# only the items required to buid the old doc with the recent parameters
#    "docs/sources": [
SOURCES_CONTENT = [
    "credits",
    "devguide",
    "gettingstarted",
    "reference",
    "userguide",
    "whatsnew",
    "index.rst",
]
# }
# REPLACE_IN_DOCS = ["_static", "_templates", "make.py", "sources/conf.py"]


class BuildOldTagDocs:
    def __init__(self, **kwargs):
        self.tag_name = kwargs.get("tag_name")
        if not self.tag_name:
            raise ValueError("Please provide a tag name.")

        self.verbose = kwargs.get("verbose")

        # Create a temporary working directory
        self.workingdir = self._create_working_docs_directory()

    def _create_working_docs_directory(self):
        """Create temporary working directory in current location."""
        TEMPDIRS.mkdir(exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="scp_", dir=TEMPDIRS))

    def _get_tag_directory_content(self, directory):
        """
        Get all files from a specific directory at a given tag.

        Parameters
        ----------
        directory : str
            Directory path to get files from.

        Returns
        -------
        list
            List of file paths in the directory at that tag.
        """
        cmd = f"git ls-tree -r {self.tag_name}:{directory}"
        try:
            result = subprocess.run(  # noqa: S603
                cmd.split(),
                capture_output=True,
                text=True,
                check=True,  # noqa: S603
            )  # noqa: S603
            files = []
            for line in result.stdout.splitlines():
                mode, type, hash, path = line.split()
                if type == "blob":  # Only include files, not subdirectories
                    files.append(path)
            return files
        except subprocess.CalledProcessError:
            return []

    def copy_docs_directory_from_tag(self):
        """Copy the docs directory from the specified tag to the temporary working directory."""
        files = self._get_tag_directory_content("docs")
        if not files:
            self.workingdir.rmdir()
            raise FileNotFoundError(
                "No files found in the docs directory at the given tag."
            )
        self._makecopy(files, "docs")

    def _makecopy(self, files, directory):
        """Copy files to the temporary directory."""
        for file in files:
            if ".DS_Store" in file or "__pycache__" in file:  # skip obvious files!
                continue
            src_path = Path(directory) / file
            if src_path.parts[1] not in SOURCES_CONTENT:
                continue
            # if
            # basedir = Path(file).parts[0]
            # if basedir in DOCS_STRUCTURE["docs"]:
            #    dst_path = self.workingdir / "docs" / file
            # elif basedir in DOCS_STRUCTURE["docs/sources"]:
            dst_path = self.workingdir / "docs" / "sources" / file
            # else:
            #    raise ValueError(f"Unknown path: {dst_path}")
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = f"git show {self.tag_name}:{src_path}"
            try:
                result = subprocess.run(  # noqa: S603
                    cmd.split(),
                    capture_output=True,
                    text=True,
                    check=True,
                    encoding="utf-8",
                    errors="replace",
                )
                dst_path.write_text(result.stdout, encoding="utf-8", errors="replace")
            except subprocess.CalledProcessError as e:
                print(f"Failed to extract {src_path} from git tag {self.tag_name}: {e}")

    # def replace_docs_members(self):
    #     """Replace specific files in the docs directory with those from the current version."""
    #     for item in REPLACE_IN_DOCS:
    #         src_path = DOCS / item
    #         dst_path = Path(self.workingdir) / "docs" / item
    #         if not src_path.exists():
    #             raise FileNotFoundError(
    #                 f"File {src_path} does not exist, so it cannot be replaced. "
    #                 "Check if the copy of the tag directory has been done properly"
    #             )
    #         if src_path.is_file():
    #             shutil.copy2(src_path, dst_path)
    #         else:
    #             self._copy_directory(src_path, dst_path)

    # def _copy_directory(self, src_dir, dst_dir):
    #     """Copy the contents of a directory."""
    #     for item in src_dir.iterdir():
    #         if item.is_dir():
    #             if (dst_dir / item.name).exists():
    #                 shutil.rmtree(dst_dir / item.name)
    #             shutil.copytree(item, dst_dir / item.name)
    #         else:
    #             shutil.copy2(item, dst_dir / item.name)
