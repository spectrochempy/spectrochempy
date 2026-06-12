# SpectroChemPy Plotting Architecture Audit

**Date:** 2026-06-12
**Audit Scope:** Plotting architecture, Matplotlib dependencies, backend extensibility
**Version:** master
**Status:** Analysis Complete

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current Architecture Map](#1-current-architecture-map)
3. [Coupling Analysis](#2-coupling-analysis)
4. [Matplotlib Dependencies Inventory](#3-matplotlib-dependencies-inventory)
5. [Extensibility Assessment](#4-extensibility-assessment)
6. [Test Coverage Inventory](#5-test-coverage-inventory)
7. [Backend-Neutral Roadmap](#6-backend-neutral-roadmap)
8. [Recommendations](#7-recommendations)

---

## Executive Summary

SpectroChemPy has recently undergone a significant refactoring of its plotting
subsystem, moving from a monolithic design tightly coupled to Matplotlib to a
modular architecture with explicit backend separation.

**Key Finding:** SpectroChemPy is already well-positioned for backend
extensibility. The architecture cleanly separates plot semantics from backend
rendering. Adding Plotly, Bokeh, or Altair backends would require minimal changes.

**Strengths:** Clean dispatcher pattern, lazy Matplotlib initialization,
stateless design, style isolation, existing Plotly integration points.

**Gaps:** 46 direct matplotlib imports, thin backend abstraction, Plotly uses
legacy architecture, no formal backend interface, Matplotlib-specific tests.

**Risk Assessment:** LOW - The architecture is sound. Adding new backends is feasible.

---

## 1. Current Architecture Map

### 1.1 Component Hierarchy

Execution flow:
```
dataset.plot(method="pen")
  -> NDDataset.plot() [thin delegator]
    -> dispatcher.plot_dataset() [backend routing + method normalization]
      -> matplotlib_backend.plot_dataset_impl() [backend implementation]
        -> plot1d.plot_pen() [plot method: data extraction + Matplotlib rendering]
```

### 1.2 Module Organization

| Module | Purpose | Status |
|--------|---------|--------|
| `plotting/dispatcher.py` | Backend routing | Current |
| `plotting/backends/matplotlib_backend.py` | Matplotlib backend | Current |
| `plotting/plot_setup.py` | Lazy Matplotlib init | Current |
| `plotting/plot1d.py` | 1D plot functions | Current |
| `plotting/plot2d.py` | 2D plot functions | Current |
| `plotting/plot3d.py` | 3D plot functions | Current |
| `plotting/multiplot.py` | Multi-panel plots | Current |
| `plotting/_style.py` | Style utilities | Current |
| `plotting/_render.py` | Rendering utilities | Current |
| `plotting/profile.py` | Plot preferences | Current |
| `core/plotters/plot*.py` | Legacy re-exports | Deprecated |
| `core/plotters/plotly.py` | Legacy Plotly | Legacy |

### 1.3 Key Architectural Decisions

- **Lazy Initialization:** ALL matplotlib imports deferred until first plot() call
- **Stateless Plotting:** NO figure/axes state stored on dataset objects
- **Style Isolation:** Preferences via context managers, NOT global rcParams
- **Backend Dispatcher:** Clean separation via registry pattern

### 1.4 Method Normalization

Legacy names mapped to canonical: stack->lines, map->contour, image->contourf
Semantic aliases (image) normalized without warning, deprecated with warning.

---

## 2. Coupling Analysis

### 2.1 Coupling Points

| Coupling Point | Type | Severity | Location |
|---------------|------|----------|----------|
| dataset.plot() delegation | Loose | LOW | nddataset.py:1590 |
| Backend dispatcher | Loose | LOW | dispatcher.py |
| Direct matplotlib imports | Tight | HIGH | 46 occurrences |
| Axes/Figure manipulation | Tight | HIGH | All plot*.py files |
| Return type (Axes) | Backend-specific | HIGH | All plot functions |

### 2.2 Dependency Direction

Only 2 levels have Matplotlib dependencies:
1. matplotlib_backend.py - Backend implementation
2. plot1d/2d/3d.py - Plot method implementations

Top-level API (NDDataset.plot(), dispatcher) is completely backend-agnostic.

---

## 3. Matplotlib Dependencies Inventory

### 3.1 Classification

- **Essential (20):** Required for Matplotlib rendering (pyplot, colors, ticker, etc.)
- **Incidental (15):** Could be abstracted (rc_context, style.context, subplots)
- **Removable (10):** Legacy code in core/plotters/

### 3.2 Count by Module

| Module | Imports | Status |
|--------|---------|--------|
| dispatcher.py | 0 | Clean |
| matplotlib_backend.py | 2 | Clean (lazy) |
| plot_setup.py | 0 | Clean (lazy) |
| plot1d.py | 3 | Matplotlib-specific |
| plot2d.py | 9 | Matplotlib-specific |
| plot3d.py | 3 | Matplotlib-specific |
| multiplot.py | 5 | Matplotlib-specific |
| _style.py | 8 | Matplotlib-specific |
| _render.py | 1 | Matplotlib-specific |
| composite/* | 3 | Matplotlib-specific |

---

## 4. Extensibility Assessment

### 4.1 Current Backend Support

| Backend | Status | Location | Integration |
|---------|--------|----------|------------|
| Matplotlib | Canonical | plotting/backends/matplotlib_backend.py | Full |
| Plotly | Legacy | core/plotters/plotly.py | Partial |
| Bokeh | None | N/A | None |
| Altair | None | N/A | None |

### 4.2 Backend Interface

Minimum interface:
```python
def plot_dataset_impl(dataset, method=None, **kwargs) -> Any:
    pass
```

### 4.3 Feasibility

- Plotly migration: LOW effort (code exists, needs refactoring)
- Bokeh/Altair: MEDIUM effort (requires abstraction layer)
- No major architectural changes needed

---

## 5. Test Coverage Inventory

### 5.1 Test Files

15+ test files in tests/test_plotting/, all Matplotlib-specific.

### 5.2 Coverage Assessment

- Strengths: Comprehensive API contract tests, good refactored/legacy coverage
- Gaps: No backend-neutral tests, no Plotly backend tests, no mock backend tests
- Coverage: MEDIUM-HIGH for Matplotlib, NONE for other backends

---

## 6. Backend-Neutral Roadmap

### Phase 0: Current State (Done)
- Dispatcher with backend registry
- Stateless plotting
- Lazy Matplotlib initialization
- Style isolation

### Phase 1: Immediate (P0)
- Migrate Plotly to new backend architecture
- Expose backend parameter in public API
- Add backend parameter to plot functions

### Phase 2: Short-Term (P1)
- Define PlotBackend ABC
- Formalize backend interface
- Add Bokeh backend

### Phase 3: Medium-Term (P2)
- Complete backend abstraction
- Add Altair backend

### Phase 4: Long-Term (P3)
- Backend auto-detection
- Interactive web rendering
- Backend performance benchmarks

---

## 7. Recommendations

### P0 - Do Now

**R0.1: Migrate Plotly to New Backend Architecture**
- Create: plotting/backends/plotly_backend.py
- Register in dispatcher
- Users call: dataset.plot(backend="plotly")
- Risk: LOW, Value: HIGH, Effort: MEDIUM

**R0.2: Expose Backend Parameter**
- Modify NDDataset.plot() to accept backend parameter
- Risk: LOW, Value: HIGH, Effort: LOW

**R0.3: Pass Backend to Plot Functions**
- Update plot1d/2d/3d to pass backend through
- Risk: LOW, Value: HIGH, Effort: LOW

### P1 - Next Release

**R1.1: Define PlotBackend ABC**
- Formal interface in plotting/backends/__init__.py
- Risk: LOW, Value: HIGH, Effort: MEDIUM

**R1.2: Add Backend Detection**
- Auto-detect from preferences
- Risk: LOW, Value: MEDIUM, Effort: LOW

### P2 - Future

**R2.1: Create Abstraction Layer**
- Backend primitives: create_figure, create_axes, line, scatter, etc.
- Risk: MEDIUM, Value: HIGH, Effort: HIGH

**R2.2: Add Bokeh Backend**
- Risk: MEDIUM, Value: HIGH, Effort: MEDIUM

---

## Appendix A: File Reference

| Component | Files | Key Lines |
|-----------|-------|-----------|
| Entry | nddataset.py | 1590-1649 |
| Dispatcher | plotting/dispatcher.py | All |
| Matplotlib backend | plotting/backends/matplotlib_backend.py | All |
| Plot setup | plotting/plot_setup.py | All |
| Plot methods | plotting/plot1d.py, plot2d.py, plot3d.py | All |
| Legacy | core/plotters/plotly.py | All |
| Tests | tests/test_plotting/*.py | All |

---

## Appendix B: Questions Answered

Q1: Map plotting stack - See Section 1.1
Q2: Semantics vs rendering - PARTIALLY separated
Q3: Matplotlib dependencies - 46 imports, classified
Q4: Extensibility - HIGH, architecture is ready
Q5: Tests - 15+ files, all Matplotlib-specific
Q6: Target architecture - Layered model feasible

---

## Conclusion

SpectroChemPy plotting architecture is already well-positioned for backend
independence. The dispatcher pattern, lazy initialization, stateless design, and
style isolation provide a solid foundation.

**Start small:** Migrate Plotly (P0), then incrementally add abstraction and
backends (P1-P3). No major architectural changes needed.

*Audit Complete*
