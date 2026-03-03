# Alternative A: Full Lazy Initialization Implementation Plan

## 🎯 **Objective**
Achieve maximum import performance (900-2550ms savings) by deferring ALL matplotlib initialization until first plot call.

## 📋 **Core Architecture Changes**

### **1. New Lazy Infrastructure in `plot_setup.py`**

#### **A. State Management System**
```python
class MPLInitState(Enum):
    NOT_INITIALIZED = "not_initialized"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    FAILED = "failed"

# Global state variables
_MPL_INIT_STATE = MPLInitState.NOT_INITIALIZED
_MPL_INIT_LOCK = threading.Lock()
_PENDING_PREFERENCE_CHANGES: Dict[str, Any] = {}
_MPL_INIT_ERROR = None
```

#### **B. Core Lazy Function**
```python
def lazy_ensure_mpl_config() -> None:
    """
    LAZY initialization of ALL matplotlib functionality.

    This is the single entry point that replaces ALL matplotlib setup
    currently scattered across app.start() and other initialization points.

    Responsibilities:
    - Ensure matplotlib is safely initialized (backend-safe)
    - Snapshot user rcParams BEFORE any modifications
    - Install SpectroChemPy matplotlib assets
    - Apply all deferred PlotPreferences changes
    - Configure LaTeX fonts
    - Apply default style ("classic" + configured style)
    - Handle thread safety and error conditions
    """
```

### **2. Preference Deferral System**

#### **A. Modified PlotPreferences Observer**
```python
# In plot_preferences.py - modify _anytrait_changed()
@observe(All)
def _anytrait_changed(self, change):
    """
    Synchronize trait changes → matplotlib.rcParams with LAZY deferral.

    If matplotlib not yet initialized, queue the change for later application.
    If matplotlib initialized, apply immediately.
    """
    if not _is_mpl_initialized():
        _defer_preference_change(change)
    else:
        _apply_preference_change_immediately(change)
```

#### **B. Preference Queue Management**
```python
def _defer_preference_change(change: Dict[str, Any]) -> None:
    """Queue a preference change until matplotlib is initialized."""

def _apply_deferred_preferences() -> None:
    """Apply all queued preference changes after matplotlib initialization."""

def _synchronize_preferences_to_rcparams() -> None:
    """Synchronize all PlotPreferences to matplotlib rcParams."""
```

### **3. Entry Point Modifications**

#### **A. Modified ndplot.py**
```python
# In NDPlot.plot() method - add at the very beginning:
def plot(self, method: str | None = None, **kwargs: Any) -> _Axes | None:
    # LAZY TRIGGER: This is the ONLY place that initializes matplotlib
    from spectrochempy.core.plotters.plot_setup import lazy_ensure_mpl_config
    lazy_ensure_mpl_config()

    # ... rest of existing plot logic
```

#### **B. Modified plot_setup.py**
```python
def ensure_spectrochempy_plot_style() -> None:
    """
    Legacy compatibility wrapper.

    Now just calls lazy_ensure_mpl_config() for backward compatibility.
    """
    lazy_ensure_mpl_config()
```

### **4. Application Startup Cleanup**

#### **A. Remove from app.start()**
Remove these sections from `application.py`:
- Lines 351: `from matplotlib import pyplot as plt`
- Lines 365-395: All rcParams snapshotting and modification
- Line 395: `plt.style.use(["classic"])`
- Lines 372-385: PlotPreferences to rcParams synchronization

#### **B. Keep in app.start()**
- Directory creation
- Configuration loading
- Logging setup

## 🔄 **Complete Initialization Flow**

### **Before Changes (Current)**
```
import spectrochempy  → app.start() → matplotlib import → rcParams modified
```

### **After Changes (Alternative A)**
```
import spectrochempy  → app.start() → NO matplotlib changes

dataset.plot() → lazy_ensure_mpl_config() → FULL matplotlib setup:
    - matplotlib import (backend-safe)
    - snapshot user rcParams
    - install assets
    - apply queued preferences
    - configure LaTeX
    - apply style
    - mark as initialized
```

## 📁 **Files to Modify**

### **Primary Changes**
1. **`src/spectrochempy/core/plotters/plot_setup.py`**
   - Add lazy infrastructure (100+ lines)
   - Implement `lazy_ensure_mpl_config()` (80+ lines)
   - Add preference queue management (60+ lines)

2. **`src/spectrochempy/application/_preferences/plot_preferences.py`**
   - Modify `_anytrait_changed()` for deferral (20 lines)
   - Add helper functions for queue management (40 lines)

3. **`src/spectrochempy/core/dataset/arraymixins/ndplot.py`**
   - Add lazy trigger in `plot()` method (3 lines)

4. **`src/spectrochempy/application/application.py`**
   - Remove matplotlib setup from `start()` (30 lines removed)

### **Secondary Changes**
5. **`src/spectrochempy/core/plotters/_mpl_setup.py`**
   - No changes needed (already lazy)

6. **`src/spectrochempy/core/plotters/_mpl_assets.py`**
   - No changes needed (already called lazily)

## 🧪 **Testing Requirements**

### **Performance Tests**
```python
def test_import_performance():
    """Verify 2.5+ second import time improvement."""

def test_first_plot_performance():
    """Verify first plot time is acceptable (<3s)."""

def test_subsequent_plot_performance():
    """Verify subsequent plots are fast (<100ms)."""
```

### **Functionality Tests**
```python
def test_lazy_initialization():
    """Verify matplotlib not modified before first plot."""

def test_preference_deferral():
    """Verify preference changes before init are applied correctly."""

def test_thread_safety():
    """Verify lazy init works in multi-threaded context."""

def test_backward_compatibility():
    """Ensure existing code continues to work."""
```

### **Integration Tests**
```python
def test_restoration_accuracy():
    """Verify restore_rcparams() works with new timing."""

def test_visual_identicality():
    """Ensure plots look identical to current implementation."""
```

## ⚡ **Performance Targets**

| Operation | Current | Target (Alternative A) |
|-----------|---------|------------------------|
| `import spectrochempy` | 900-2550ms | 0ms |
| First `dataset.plot()` | N/A | 950-2650ms |
| Subsequent plots | 50-100ms | 50-100ms |
| Total to first plot | 900-2550ms | 950-2650ms |

## 🔍 **Risk Analysis & Mitigations**

### **Implementation Risks**
1. **Complex State Management**
   - **Mitigation**: Clear enum states, comprehensive logging

2. **Preference Synchronization Issues**
   - **Mitigation**: Extensive testing, fallback mechanisms

3. **Thread Safety Problems**
   - **Mitigation**: Proper locking, idempotent design

4. **Backward Compatibility Issues**
   - **Mitigation**: Wrapper functions, thorough testing

### **Operational Risks**
1. **Error Handling Complexity**
   - **Mitigation**: Try-catch blocks, graceful degradation

2. **Debugging Difficulty**
   - **Mitigation**: Detailed debug logging, state inspection

## 📝 **Implementation Checklist**

### **Phase 1: Core Infrastructure**
- [ ] Implement MPLInitState enum and global state
- [ ] Add thread-safe lazy_ensure_mpl_config() function
- [ ] Create preference deferral system
- [ ] Add error handling and logging

### **Phase 2: Integration**
- [ ] Modify ndplot.py to trigger lazy init
- [ ] Update PlotPreferences observer for deferral
- [ ] Remove matplotlib setup from app.start()
- [ ] Add backward compatibility wrappers

### **Phase 3: Testing & Validation**
- [ ] Performance benchmarking
- [ ] Functionality testing
- [ ] Thread safety validation
- [ ] Visual plot comparison
- [ ] Backward compatibility verification

### **Phase 4: Documentation**
- [ ] Update API documentation
- [ ] Add migration guide
- [ ] Document performance improvements
- [ ] Update troubleshooting guide

## 🎯 **Success Criteria**

✅ **Import time reduced from 900-2550ms to 0ms**
✅ **All existing tests pass**
✅ **Restoration works perfectly**
✅ **No visual changes to plots**
✅ **Thread-safe initialization**
✅ **Backward compatibility maintained**

This comprehensive plan for Alternative A will achieve the maximum performance improvement while maintaining all existing functionality and safety guarantees.
