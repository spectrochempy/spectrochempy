# Alternative A: Technical Implementation Details

## 🔧 **Core Implementation Code**

### **1. Enhanced plot_setup.py - Lazy Infrastructure**

```python
# Add to imports (after existing imports)
from enum import Enum
from typing import Any, Dict, Optional
import threading

# Replace existing _USER_RCPARAMS section with:

# -----------------------------------------------------------------------------
# Lazy initialization state management
# -----------------------------------------------------------------------------

class MPLInitState(Enum):
    """Matplotlib initialization state enumeration."""
    NOT_INITIALIZED = "not_initialized"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    FAILED = "failed"

# Global state for lazy initialization - NO imports at module level!
_MPL_INIT_STATE = MPLInitState.NOT_INITIALIZED
_MPL_INIT_LOCK = threading.Lock()
_PENDING_PREFERENCE_CHANGES: Dict[str, Any] = {}
_MPL_INIT_ERROR: Optional[Exception] = None

# -----------------------------------------------------------------------------
# Internal storage for the user's original rcParams
# -----------------------------------------------------------------------------

_USER_RCPARAMS = None

def _is_mpl_initialized() -> bool:
    """Check if matplotlib has been initialized by SpectroChemPy."""
    return _MPL_INIT_STATE == MPLInitState.INITIALIZED

def _is_mpl_initializing() -> bool:
    """Check if matplotlib is currently being initialized."""
    return _MPL_INIT_STATE == MPLInitState.INITIALIZING

def _set_mpl_state(state: MPLInitState) -> None:
    """Set the matplotlib initialization state."""
    global _MPL_INIT_STATE
    _MPL_INIT_STATE = state

def _get_mpl_state() -> MPLInitState:
    """Get the current matplotlib initialization state."""
    return _MPL_INIT_STATE

# -----------------------------------------------------------------------------
# Lazy preference deferral system
# -----------------------------------------------------------------------------

def _defer_preference_change(change: Dict[str, Any]) -> None:
    """Queue a preference change until matplotlib is initialized."""
    change_key = f"{change.name}:{change.new}"
    _PENDING_PREFERENCE_CHANGES[change_key] = change
    debug_(f"Deferred preference change: {change.name} = {change.new}")

def _apply_deferred_preferences() -> None:
    """Apply all queued preference changes after matplotlib initialization."""
    if not _PENDING_PREFERENCE_CHANGES:
        return

    debug_(f"Applying {len(_PENDING_PREFERENCE_CHANGES)} deferred preference changes")

    # Import here to avoid import-time matplotlib dependencies
    from spectrochempy.application.preferences import preferences
    plot_prefs = preferences.get("plot", None)

    if plot_prefs:
        for change_key, change in _PENDING_PREFERENCE_CHANGES.items():
            try:
                # Apply the change directly to rcParams
                plot_prefs._apply_preference_change_immediately(change)
            except Exception as e:
                warning_(f"Failed to apply deferred preference {change_key}: {e}")

    # Clear the queue
    _PENDING_PREFERENCE_CHANGES.clear()

def _synchronize_preferences_to_rcparams() -> None:
    """Synchronize all PlotPreferences to matplotlib rcParams."""
    from spectrochempy.application.preferences import preferences
    plot_prefs = preferences.get("plot", None)

    if not plot_prefs:
        return

    # Import matplotlib here (lazy)
    import matplotlib as mpl

    # Force update all rcParams from preferences
    for rckey in mpl.rcParams:
        key = rckey.replace("_", "__").replace(".", "_").replace("-", "___")
        try:
            value = getattr(plot_prefs, key)
            if value is not None:
                mpl.rcParams[rckey] = value
        except (ValueError, AttributeError):
            pass  # Graceful handling
```

### **2. Core Lazy Function Implementation**

```python
def lazy_ensure_mpl_config() -> None:
    """
    LAZY initialization of ALL matplotlib functionality.

    This is the single entry point that replaces ALL matplotlib setup
    currently scattered across app.start() and other initialization points.

    This function is:
    - Thread-safe
    - Idempotent
    - Comprehensive (handles ALL matplotlib setup)
    - Error-resilient
    """
    # Fast path: already initialized
    if _is_mpl_initialized():
        return

    # Thread safety: ensure only one thread initializes
    with _MPL_INIT_LOCK:
        # Double-check pattern
        if _is_mpl_initialized():
            return

        if _is_mpl_initializing():
            # Another thread is initializing, wait for completion
            while _is_mpl_initializing():
                threading.Event().wait(0.01)  # 10ms polling
            return

        # Mark as initializing
        _set_mpl_state(MPLInitState.INITIALIZING)

        try:
            _perform_lazy_mpl_initialization()
            _set_mpl_state(MPLInitState.INITIALIZED)
            debug_("Lazy matplotlib initialization completed successfully")

        except Exception as e:
            _set_mpl_state(MPLInitState.FAILED)
            global _MPL_INIT_ERROR
            _MPL_INIT_ERROR = e
            error_(e, "Failed to initialize matplotlib lazily")
            # Re-raise to let caller know initialization failed
            raise

def _perform_lazy_mpl_initialization() -> None:
    """
    Perform the actual matplotlib initialization.

    This contains ALL the matplotlib setup logic that was previously
    scattered across app.start(), ensure_spectrochempy_plot_style(), etc.
    """
    # ------------------------------------------------------------------
    # 1. Ensure matplotlib is initialized safely (backend-safe)
    # ------------------------------------------------------------------
    from spectrochempy.core.plotters._mpl_setup import ensure_mpl_setup
    ensure_mpl_setup()

    # ------------------------------------------------------------------
    # 2. Snapshot user rcParams BEFORE touching anything
    # ------------------------------------------------------------------
    _snapshot_user_rcparams()

    # ------------------------------------------------------------------
    # 3. High-level imports (safe AFTER ensure_mpl_setup)
    # ------------------------------------------------------------------
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # ------------------------------------------------------------------
    # 4. Install matplotlib assets (best effort)
    # ------------------------------------------------------------------
    from spectrochempy.core.plotters._mpl_assets import ensure_mpl_assets_installed
    with contextlib.suppress(Exception):
        ensure_mpl_assets_installed()

    # ------------------------------------------------------------------
    # 5. Apply all deferred preference changes
    # ------------------------------------------------------------------
    _apply_deferred_preferences()

    # ------------------------------------------------------------------
    # 6. Synchronize current preferences to rcParams
    # ------------------------------------------------------------------
    _synchronize_preferences_to_rcparams()

    # ------------------------------------------------------------------
    # 7. LaTeX font configuration
    # ------------------------------------------------------------------
    from spectrochempy.application.preferences import preferences
    plot_prefs = preferences.get("plot", None)
    if plot_prefs:
        plot_prefs.set_latex_font(plot_prefs.font_family)

    # ------------------------------------------------------------------
    # 8. Apply default style (classic + configured style)
    # ------------------------------------------------------------------
    plt.style.use(["classic"])

    if plot_prefs and plot_prefs.style:
        if isinstance(plot_prefs.style, str):
            plt.style.use([plot_prefs.style])
        else:
            plt.style.use(plot_prefs.style)
```

### **3. Modified PlotPreferences Observer**

```python
# In plot_preferences.py - replace existing _anytrait_changed method

@observe(All)
def _anytrait_changed(self, change):
    """
    Synchronize trait changes → matplotlib.rcParams with LAZY deferral.

    This method now handles both immediate and deferred preference changes
    depending on whether matplotlib has been initialized.
    """
    from spectrochempy.core.plotters.plot_setup import _is_mpl_initialized, _defer_preference_change

    # Queue the change if matplotlib not yet initialized
    if not _is_mpl_initialized():
        _defer_preference_change(change)
        super()._anytrait_changed(change)
        return

    # Apply immediately if matplotlib is already initialized
    self._apply_preference_change_immediately(change)

def _apply_preference_change_immediately(self, change):
    """
    Apply a preference change immediately to matplotlib rcParams.

    This contains the original logic from _anytrait_changed, separated
    for clarity and reuse in deferred preference application.
    """
    import matplotlib as mpl

    if change.name in self.trait_names(config=True):
        key = self.to_rc_key(change.name)
        if key in mpl.rcParams:
            try:
                mpl.rcParams[key] = change.new
            except ValueError:
                mpl.rcParams[key] = change.new.replace("'", "")

        # Special handling for font size cascading
        if key == "font.size":
            mpl.rcParams["legend.fontsize"] = int(change.new * 0.8)
            mpl.rcParams["xtick.labelsize"] = int(change.new)
            mpl.rcParams["ytick.labelsize"] = int(change.new)
            mpl.rcParams["axes.labelsize"] = int(change.new)

        # Special handling for font family LaTeX configuration
        if key == "font.family":
            self.set_latex_font(change.new)

    super()._anytrait_changed(change)
```

### **4. Modified ndplot.py Entry Point**

```python
# In ndplot.py - modify the plot method

@docprocess.get_sections(
    base="plot",
    sections=["Parameters", "Other Parameters", "Returns"],
)
@docprocess.dedent
def plot(self, method: str | None = None, **kwargs: Any) -> _Axes | None:
    """
    Plot the dataset using the specified method.

    [Keep existing docstring...]
    """

    # 🚀 LAZY TRIGGER: This is the ONLY place that initializes matplotlib
    # ALL matplotlib setup happens here on the first plot() call
    from spectrochempy.core.plotters.plot_setup import lazy_ensure_mpl_config
    lazy_ensure_mpl_config()

    # Remove the old ensure_spectrochempy_plot_style() call since
    # lazy_ensure_mpl_config() handles everything now

    show = kwargs.pop("show", True)

    # --- Default plotting method ---
    if method is None:
        if self._squeeze_ndim == 1:
            method = "pen"
        elif self._squeeze_ndim == 2:
            method = "stack"
        elif self._squeeze_ndim == 3:
            method = "surface"

    _plotter = getattr(self, f"plot_{method.replace('+', '_')}", None)
    if _plotter is None:
        error_(
            NameError,
            f"The specified plotter for method `{method}` was not found!",
        )
        raise OSError

    ax = _plotter(**kwargs)

    if show:
        mpl_show()

    return ax
```

### **5. Application Startup Cleanup**

```python
# In application.py - modify the start() method

def start(self):
    """
    Start the SpectroChemPy application.

    MATPLOTLIB SETUP REMOVED: All matplotlib initialization is now lazy
    and happens on the first plot() call via lazy_ensure_mpl_config().

    This provides dramatic import performance improvements.
    """
    # ... keep existing setup until line 364 ...

    # Get preferences from the config file and init everything
    self._init_all_preferences()

    # ❌ REMOVED: Snapshot user rcParams - now done lazily
    # from spectrochempy.core.plotters.plot_setup import _snapshot_user_rcparams
    # _snapshot_user_rcparams()

    # ❌ REMOVED: Force update of rcParams - now done lazily
    # import matplotlib as mpl
    # for rckey in mpl.rcParams:
    #     key = rckey.replace("_", "__").replace(".", "_").replace("-", "___")
    #     try:
    #         mpl.rcParams[rckey] = getattr(self.plot_preferences, key)
    #     except ValueError:
    #         mpl.rcParams[rckey] = getattr(self.plot_preferences, key).replace(
    #             "'",
    #             "",
    #         )
    #     except AttributeError:
    #         pass

    # ❌ REMOVED: LaTeX font configuration - now done lazily
    # self.plot_preferences.set_latex_font(self.plot_preferences.font_family)

    # Eventually write default config file - KEEP THIS
    self.make_default_config_file()

    # ❌ REMOVED: Set default style - now done lazily
    # plt.style.use(["classic"])

    # ... keep the rest of the method ...
```

### **6. Backward Compatibility Wrapper**

```python
# In plot_setup.py - modify existing ensure_spectrochempy_plot_style()

def ensure_spectrochempy_plot_style() -> None:
    """
    Legacy compatibility wrapper.

    Previously this function handled all matplotlib setup.
    Now it delegates to the new lazy system for backward compatibility.

    This ensures existing code that calls this function continues to work.
    """
    lazy_ensure_mpl_config()
```

## 🧪 **Testing Implementation**

### **Performance Test**
```python
def test_lazy_import_performance():
    """Verify that import doesn't initialize matplotlib."""
    import time
    import sys

    # Fresh import in subprocess
    start = time.time()
    import spectrochempy
    import_time = time.time() - start

    # Should be < 100ms (no matplotlib initialization)
    assert import_time < 0.1, f"Import too slow: {import_time:.3f}s"

    # matplotlib should not be imported yet
    assert 'matplotlib.pyplot' not in sys.modules
    assert 'matplotlib' not in sys.modules
```

### **Functionality Test**
```python
def test_lazy_plot_initialization():
    """Verify that first plot initializes matplotlib correctly."""
    from spectrochempy import NDDataset
    import numpy as np

    # Create dataset and plot - should trigger lazy init
    data = np.random.rand(100)
    dataset = NDDataset(data)

    # This should work without errors
    ax = dataset.plot()
    assert ax is not None

    # matplotlib should now be initialized
    from spectrochempy.core.plotters.plot_setup import _is_mpl_initialized
    assert _is_mpl_initialized()
```

This technical implementation provides the complete code changes needed for Alternative A, achieving maximum performance improvement while maintaining all existing functionality.
