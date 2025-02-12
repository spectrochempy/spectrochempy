# Utility for docs generation
# ruff: noqa: T201,S603

# Import required modules
import shlex
from pathlib import Path
from subprocess import PIPE
from subprocess import STDOUT
from subprocess import run


class sh:
    """
    Shell command execution utility.

    This class provides a convenient interface to run shell commands safely
    by wrapping subprocess.run().
    """

    def __getattr__(self, command: str) -> "_ExecCommand":
        """
        Create a new command executor for the given command.

        Parameters
        ----------
        command : str
            The shell command to execute

        Returns
        -------
        _ExecCommand
            A command executor instance
        """
        return _ExecCommand(command)

    def __call__(
        self, script: str | Path, silent: bool = False, capture_stderr: bool = False
    ) -> str | None:
        """
        Run a shell script safely.

        Parameters
        ----------
        script : str or Path
            The script command to execute
        silent : bool, optional
            If True, suppress output to stdout
        capture_stderr : bool, optional
            If True, capture stderr separately from stdout

        Returns
        -------
        str or None
            Command output if any. If capture_stderr is True, returns stdout

        Raises
        ------
        subprocess.CalledProcessError
            If the command fails to execute
        """
        if isinstance(script, Path):
            script = str(script)

        try:
            safe_script = shlex.split(script)
            stderr_pipe = PIPE if capture_stderr else STDOUT
            proc = run(
                safe_script,
                text=True,
                stdout=PIPE,
                stderr=stderr_pipe,
                check=True,  # Raise CalledProcessError on non-zero exit
            )

            if not silent:
                if proc.stdout:
                    print(proc.stdout)
                if capture_stderr and proc.stderr:
                    print(proc.stderr)

            return proc.stdout

        except Exception as e:
            if not silent:
                print(f"Error executing command: {e}")
            raise


class _ExecCommand:
    """
    Shell command executor with arguments.

    This class handles the execution of shell commands with their arguments
    in a safe manner.
    """

    def __init__(self, command: str):
        """
        Initialize with base command.

        Parameters
        ----------
        command : str
            The base command to execute
        """
        self.commands: list[str] = [command]

    def __call__(
        self, *args, silent: bool = False, capture_stderr: bool = False
    ) -> str | None:
        """
        Execute the command with given arguments.

        Parameters
        ----------
        *args : Any
            Command arguments
        silent : bool, optional
            If True, suppress output to stdout
        capture_stderr : bool, optional
            If True, capture stderr separately from stdout

        Returns
        -------
        str or None
            Command output if any

        Raises
        ------
        subprocess.CalledProcessError
            If the command fails to execute
        """
        args = [str(arg) for arg in args]  # Convert all args to strings
        self.commands.extend(args)

        try:
            safe_command = shlex.split(" ".join(self.commands))
            stderr_pipe = PIPE if capture_stderr else STDOUT
            proc = run(
                safe_command,
                text=True,
                stdout=PIPE,
                stderr=stderr_pipe,
                check=True,
            )

            if not silent:
                if proc.stdout:
                    print(proc.stdout)
                if capture_stderr and proc.stderr:
                    print(proc.stderr)
            return proc.stdout

        except Exception as e:
            if not silent:
                print(f"Error executing command: {e}")
            raise


# Create singleton instance
sh = sh()
