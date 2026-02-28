# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Plot profile management for SpectroChemPy.

This module provides a named profile system for controlling default plotting
settings. Profiles can include SpectroChemPy-specific defaults and matplotlib
style sheet references.

Key features:
- Named profiles (e.g., "default", "scp_paper", "scp_talk")
- Built-in profiles stored in the package
- User profiles stored in ~/.spectrochempy/config/plot_profiles/
- Lazy initialization (no I/O until first plot)
- Full backward compatibility with existing preferences

Usage:
    >>> import spectrochempy as scp
    >>> scp.set_plot_profile("scp_paper")
    >>> dataset.plot()  # Uses paper profile settings
"""

import json
import warnings
from pathlib import Path
from typing import Any

# ======================================================================================
# Module-level state
# ======================================================================================

_profile_manager = None


# ======================================================================================
# PlotProfileManager class
# ======================================================================================


class PlotProfileManager:
    """
    Manages named plot profiles with lazy initialization.

    Profiles control default PlotPreferences values for plotting operations.
    This class provides a unified interface for managing plotting profiles,
    including built-in profiles and user-defined profiles.

    Attributes
    ----------
    _initialized : bool
        Whether the manager has been initialized.
    _active_profile : str
        Name of the currently active profile.
    _profile_cache : dict
        Cache of loaded profile data.
    """

    BUILTIN_PROFILES = frozenset(["default", "scp_paper", "scp_talk", "scp_grayscale"])

    def __init__(self):
        """Initialize the profile manager."""
        self._initialized = False
        self._active_profile = "default"
        self._profile_applied = (
            False  # Track if profile has been applied to preferences
        )
        self._profile_cache = {}
        self._config_dir = None
        self._profiles_dir = None
        self._known_profiles_cache = None

    @property
    def config_dir(self) -> Path:
        """Get the configuration directory."""
        if self._config_dir is None:
            from spectrochempy.application.application import get_config_dir

            self._config_dir = get_config_dir()
        return self._config_dir

    @property
    def profiles_dir(self) -> Path:
        """Get the user profiles directory."""
        if self._profiles_dir is None:
            self._profiles_dir = self.config_dir / "plot_profiles"
        return self._profiles_dir

    @property
    def is_initialized(self) -> bool:
        """Check if the manager has been initialized."""
        return self._initialized

    @property
    def active_profile(self) -> str:
        """Get the name of the active profile."""
        return self._active_profile

    def ensure_initialized(self) -> None:
        """Lazy initialization - loads profiles on first call."""
        if self._initialized:
            return

        self._ensure_profiles_directory()
        self._migrate_legacy_preferences()
        self._load_active_profile()
        self._initialized = True

        # NOTE: We do NOT apply profile defaults here automatically.
        # Profile defaults are applied ONLY when:
        # 1. User explicitly calls set_plot_profile()
        # 2. First plot() call, ONLY if user hasn't modified any preferences
        # This ensures user modifications to preferences are preserved.

    def set_plot_profile(self, name: str) -> None:
        """Set the active plot profile by name."""
        self.ensure_initialized()

        # Get list of available profiles
        available = self._get_all_profile_names()
        if name not in available:
            raise ValueError(
                f"Profile '{name}' not found. Available profiles: {available}"
            )

        self._active_profile = name
        self._save_active_profile_marker(name)

        # Re-apply profile defaults when user explicitly switches
        self._profile_applied = True
        self.load_profile_to_preferences(name)

    def get_plot_profile(self) -> str:
        """Get the name of the currently active plot profile."""
        self.ensure_initialized()
        return self._active_profile

    def list_plot_profiles(self) -> list[str]:
        """List all available plot profiles."""
        self.ensure_initialized()
        return self._get_all_profile_names()

    def save_plot_profile(self, name: str) -> None:
        """Save current PlotPreferences as a named profile."""
        self.ensure_initialized()

        if name in self.BUILTIN_PROFILES:
            raise ValueError(
                f"Cannot save profile with built-in name '{name}'. "
                f"Use a different name."
            )

        prefs = self._get_plot_preferences()

        profile_data = {
            "name": name,
            "description": f"User-defined profile '{name}'",
            "mpl_style": getattr(prefs, "style", "scpy"),
        }

        for trait_name in prefs.traits():
            if trait_name.startswith("_"):
                continue
            try:
                value = getattr(prefs, trait_name)
                if self._is_serializable(value):
                    profile_data[trait_name] = value
            except Exception:
                pass

        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        profile_file = self.profiles_dir / f"{name}.json"
        with open(profile_file, "w") as f:
            json.dump(profile_data, f, indent=4)

        # Clear cache
        self._known_profiles_cache = None

    def delete_plot_profile(self, name: str) -> None:
        """Delete a user-defined profile."""
        self.ensure_initialized()

        if name in self.BUILTIN_PROFILES:
            raise ValueError(f"Cannot delete built-in profile '{name}'")

        profile_file = self.profiles_dir / f"{name}.json"
        if not profile_file.exists():
            raise ValueError(f"Profile '{name}' not found")

        profile_file.unlink()

        if self._active_profile == name:
            self.set_plot_profile("default")

        self._known_profiles_cache = None

    def reset_plot_profile(self, name: str | None = None) -> None:
        """Reset a profile to its default values."""
        self.ensure_initialized()

        target = name or self._active_profile

        if target in self.BUILTIN_PROFILES:
            raise ValueError(f"Cannot reset built-in profile '{target}'")

        profile_file = self.profiles_dir / f"{target}.json"
        if not profile_file.exists():
            raise ValueError(f"Profile '{target}' not found")

        profile_file.unlink()
        self._known_profiles_cache = None

    def load_profile_to_preferences(self, name: str) -> None:
        """Load a profile's values into PlotPreferences traitlets."""
        profile_data = self.get_profile_data(name)

        prefs = self._get_plot_preferences()

        original_autosave = getattr(prefs, "_auto_save", True)
        prefs._auto_save = False

        try:
            mpl_style = profile_data.get("mpl_style", "scpy")

            for key, value in profile_data.items():
                if key in ("name", "description", "mpl_style"):
                    continue

                if hasattr(prefs, key):
                    try:
                        setattr(prefs, key, value)
                    except Exception:
                        pass

            if hasattr(prefs, "style"):
                try:
                    prefs.style = mpl_style
                except Exception:
                    pass

        finally:
            prefs._auto_save = original_autosave

    def get_profile_data(self, name: str) -> dict:
        """Get raw profile data without applying to preferences."""
        if name in self._profile_cache:
            return self._profile_cache[name]

        profile_data = self._load_profile_json(name)
        self._profile_cache[name] = profile_data
        return profile_data

    def get_mpl_style(self) -> str:
        """Get the matplotlib style name for the active profile."""
        self.ensure_initialized()
        profile_data = self.get_profile_data(self._active_profile)
        return profile_data.get("mpl_style", "scpy")

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _ensure_profiles_directory(self) -> None:
        """Ensure the user profiles directory exists."""
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    def _get_plot_preferences(self):
        """Get the PlotPreferences instance."""
        from spectrochempy.application.application import _get_environment
        from spectrochempy.application.application import app

        if not hasattr(app, "plot_preferences"):
            _get_environment()

        return app.plot_preferences

    def _get_all_profile_names(self) -> list[str]:
        """Get list of all profile names (built-in + user)."""
        if self._known_profiles_cache is not None:
            return self._known_profiles_cache

        profiles = set(self.BUILTIN_PROFILES)

        if self.profiles_dir.exists():
            for f in self.profiles_dir.glob("*.json"):
                if f.name == "active_profile.json":
                    continue
                name = f.stem
                if name not in self.BUILTIN_PROFILES:
                    profiles.add(name)

        self._known_profiles_cache = sorted(profiles)
        return self._known_profiles_cache

    def _load_profile_json(self, name: str) -> dict:
        """Load profile JSON from disk."""
        if name in self.BUILTIN_PROFILES:
            profile_data = self._load_builtin_profile(name)
            if profile_data is not None:
                return profile_data

        profile_file = self.profiles_dir / f"{name}.json"
        if profile_file.exists():
            with open(profile_file) as f:
                return json.load(f)

        raise ValueError(f"Profile '{name}' not found")

    def _load_builtin_profile(self, name: str) -> dict | None:
        """Load a built-in profile from the package."""
        from spectrochempy import __file__ as scp_file

        package_dir = Path(scp_file).parent
        profile_file = package_dir / "plotting" / "profiles" / f"{name}.json"

        if profile_file.exists():
            with open(profile_file) as f:
                return json.load(f)

        return None

    def _load_active_profile(self) -> None:
        """Load the active profile from the marker file."""
        active_file = self.profiles_dir / "active_profile.json"

        if active_file.exists():
            try:
                with open(active_file) as f:
                    data = json.load(f)
                    name = data.get("name", "default")

                    if name in self._get_all_profile_names():
                        self._active_profile = name
            except (json.JSONDecodeError, OSError):
                pass

    def _save_active_profile_marker(self, name: str) -> None:
        """Save the active profile marker to disk."""
        active_file = self.profiles_dir / "active_profile.json"

        with open(active_file, "w") as f:
            json.dump({"name": name}, f)

    def _migrate_legacy_preferences(self) -> None:
        """Migrate existing PlotPreferences.json to profiles system if needed."""
        legacy_file = self.config_dir / "PlotPreferences.json"

        default_profile = self.profiles_dir / "default.json"
        if default_profile.exists() and legacy_file.exists():
            import shutil

            backup_file = legacy_file.with_suffix(".json.bak")
            if not backup_file.exists():
                shutil.move(str(legacy_file), str(backup_file))
            return

        if legacy_file.exists() and not default_profile.exists():
            import shutil

            try:
                with open(legacy_file) as f:
                    legacy_data = json.load(f)

                plot_prefs = legacy_data.get("PlotPreferences", {})

                if "mpl_style" not in plot_prefs:
                    plot_prefs["mpl_style"] = "scpy"

                plot_prefs["name"] = "default"
                plot_prefs["description"] = "Default profile (migrated from legacy)"

                with open(default_profile, "w") as f:
                    json.dump(plot_prefs, f, indent=4)

                self._save_active_profile_marker("default")

                backup_file = legacy_file.with_suffix(".json.bak")
                shutil.move(str(legacy_file), str(backup_file))

            except (json.JSONDecodeError, OSError) as e:
                warnings.warn(f"Failed to migrate legacy preferences: {e}")

    def _is_serializable(self, value: Any) -> bool:
        """Check if a value is JSON serializable."""
        if value is None:
            return True
        if isinstance(value, (str, int, float, bool)):
            return True
        if isinstance(value, (list, tuple)):
            return all(self._is_serializable(v) for v in value)
        if isinstance(value, dict):
            return all(
                self._is_serializable(k) and self._is_serializable(v)
                for k, v in value.items()
            )
        return False


# ======================================================================================
# Singleton accessor
# ======================================================================================


def _get_profile_manager() -> PlotProfileManager:
    """Get the singleton PlotProfileManager instance."""
    global _profile_manager
    if _profile_manager is None:
        _profile_manager = PlotProfileManager()
    return _profile_manager


# ======================================================================================
# Public API functions
# ======================================================================================


def ensure_plot_profile_loaded() -> None:
    """
    Lazy initialization - ensures profile system is ready.

    Called at the beginning of any plot() function.

    NOTE: Profile defaults are NOT auto-applied.
    User must explicitly call set_plot_profile() to apply a profile.
    This ensures backward compatibility - user modifications to preferences
    are never overwritten by profile auto-loading.
    """
    manager = _get_profile_manager()
    manager.ensure_initialized()
    # NO auto-application - profile must be explicit


def set_plot_profile(name: str) -> None:
    """Set the active plot profile by name."""
    manager = _get_profile_manager()
    manager.set_plot_profile(name)


def get_plot_profile() -> str:
    """Get the name of the currently active plot profile."""
    manager = _get_profile_manager()
    return manager.get_plot_profile()


def list_plot_profiles() -> list[str]:
    """List all available plot profiles."""
    manager = _get_profile_manager()
    return manager.list_plot_profiles()


def save_plot_profile(name: str) -> None:
    """Save current PlotPreferences as a named profile."""
    manager = _get_profile_manager()
    manager.save_plot_profile(name)


def delete_plot_profile(name: str) -> None:
    """Delete a user-defined profile."""
    manager = _get_profile_manager()
    manager.delete_plot_profile(name)


def get_mpl_style() -> str:
    """Get the matplotlib style name for the active profile."""
    manager = _get_profile_manager()
    return manager.get_mpl_style()


# ======================================================================================
# Module initialization
# ======================================================================================

__all__ = [
    "PlotProfileManager",
    "ensure_plot_profile_loaded",
    "set_plot_profile",
    "get_plot_profile",
    "list_plot_profiles",
    "save_plot_profile",
    "delete_plot_profile",
    "get_mpl_style",
]
