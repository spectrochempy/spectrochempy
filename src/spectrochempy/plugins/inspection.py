# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Lightweight user-facing plugin inspection."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata

from spectrochempy.plugins.features import OFFICIAL_PLUGINS
from spectrochempy.plugins.manager import ENTRY_POINT_GROUP


@dataclass(frozen=True)
class PluginStatus:
    """Display-friendly status for one plugin."""

    name: str
    title: str
    status: str
    namespace: str = ""
    package: str = ""
    version: str = ""
    entry_point: str = ""


class PluginInspectionResult:
    """Human-readable snapshot returned by ``scp.plugins()``."""

    def __init__(
        self,
        official: list[PluginStatus],
        discovered: list[PluginStatus],
        verbose: bool = False,
    ) -> None:
        self.official = official
        self.discovered = discovered
        self.verbose = verbose

    @property
    def namespaces(self) -> list[str]:
        """Return available official plugin namespaces."""
        return [
            f"scp.{plugin.namespace}"
            for plugin in self.official
            if plugin.status == "installed" and plugin.namespace
        ]

    def __str__(self) -> str:
        lines = ["Installed official plugins", "--------------------------"]
        width = max([len(plugin.title) for plugin in self.official] + [0])
        for plugin in self.official:
            line = f"{plugin.title:<{width}}  {plugin.status}"
            if self.verbose:
                details = []
                if plugin.version:
                    details.append(plugin.version)
                if plugin.package:
                    details.append(plugin.package)
                if details:
                    line = f"{line}  ({', '.join(details)})"
            lines.append(line)

        lines.extend(["", "Available namespaces", "--------------------"])
        if self.namespaces:
            lines.extend(self.namespaces)
        else:
            lines.append("none")

        extra = [
            plugin
            for plugin in self.discovered
            if plugin.name not in {official.name for official in self.official}
        ]
        if self.verbose and extra:
            lines.extend(["", "Other discovered plugins", "------------------------"])
            width = max(len(plugin.name) for plugin in extra)
            for plugin in extra:
                line = f"{plugin.name:<{width}}  discovered"
                if plugin.entry_point:
                    line = f"{line}  ({plugin.entry_point})"
                lines.append(line)

        return "\n".join(lines)

    def __repr__(self) -> str:
        return str(self)

    def _repr_html_(self) -> str:
        rows = "\n".join(
            "<tr>"
            f"<td>{plugin.title}</td>"
            f"<td>{plugin.status}</td>"
            f"<td>{plugin.version or ''}</td>"
            f"<td>{f'scp.{plugin.namespace}' if plugin.namespace else ''}</td>"
            "</tr>"
            for plugin in self.official
        )
        return (
            "<table>"
            "<thead><tr><th>Plugin</th><th>Status</th><th>Version</th>"
            "<th>Namespace</th></tr></thead>"
            f"<tbody>{rows}</tbody>"
            "</table>"
        )


def inspect_plugins(verbose: bool = False) -> PluginInspectionResult:
    """
    Return a lightweight snapshot of installed/discovered plugins.

    This function reads package metadata and entry-point declarations only. It does
    not load plugin modules and therefore does not import optional heavy
    dependencies.
    """

    entry_points = _plugin_entry_points()
    official = [
        _official_plugin_status(name, info, entry_points)
        for name, info in OFFICIAL_PLUGINS.items()
    ]
    discovered = [
        PluginStatus(
            name=ep.name,
            title=ep.name,
            status="discovered",
            entry_point=ep.value,
        )
        for ep in entry_points
    ]
    return PluginInspectionResult(official, discovered, verbose=verbose)


def _plugin_entry_points() -> list[metadata.EntryPoint]:
    try:
        return list(metadata.entry_points(group=ENTRY_POINT_GROUP))
    except TypeError:  # pragma: no cover - compatibility with older importlib APIs
        return list(metadata.entry_points().get(ENTRY_POINT_GROUP, ()))


def _official_plugin_status(
    name: str,
    info: dict[str, str],
    entry_points: list[metadata.EntryPoint],
) -> PluginStatus:
    package = info["package"]
    namespace = info["namespace"]
    version = _distribution_version(package)
    entry_point = next((ep for ep in entry_points if ep.name == namespace), None)
    installed = version is not None or entry_point is not None
    return PluginStatus(
        name=name,
        title=info["title"],
        status="installed" if installed else "missing",
        namespace=namespace if installed else "",
        package=package,
        version=version or "",
        entry_point=entry_point.value if entry_point else "",
    )


def _distribution_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None
