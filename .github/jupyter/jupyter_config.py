"""
Jupyter configuration used only for documentation CI builds.

This keeps kernel traffic on local IPC sockets during docs execution on
GitHub-hosted Linux runners, which avoids repeated local TCP transport
warnings from IPython kernel startup.
"""

from traitlets.config import get_config

c = get_config()
c.KernelManager.transport = "ipc"
c.AsyncKernelManager.transport = "ipc"
