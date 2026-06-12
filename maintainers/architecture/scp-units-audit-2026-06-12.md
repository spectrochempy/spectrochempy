# SpectroChemPy Units and Quantity Semantics Audit

**Date:** 2026-06-12
**Audit Scope:** Unit system and quantity propagation across SpectroChemPy
**Version:** master
**Status:** Analysis Complete

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current Implementation Map](#1-current-implementation-map)
3. [Test Coverage Inventory](#2-test-coverage-inventory)
4. [Inconsistent or Underspecified Behaviors](#3-inconsistent-or-underspecified-behaviors)
5. [Conservative v1 Unit Contract Proposal](#4-conservative-v1-unit-contract-proposal)
6. [Recommendations](#5-recommendations)

---

## Executive Summary

SpectroChemPy uses Pint for unit handling with a custom UnitRegistry (`ur`) that defines spectroscopy-specific units (absorbance, transmittance, Kubelka_Munk, ppm). Units are stored at multiple levels: `NDDataset.units` for data values and `Coord.units` for coordinate axes. The `NDMath` mixin provides arithmetic operation support with unit propagation via the `_op()` method chain.

**Strengths:** Comprehensive Pint integration, consistent unit propagation in most arithmetic operations, explicit unit conversion, good spectroscopy-specific conversions.

**Gaps:** Coordinate units ignored in arithmetic, silent unit drops in comparisons, inconsistent Quantity vs NDDataset behavior, limited test coverage.

**Risk Assessment:** MEDIUM - Works for common cases but subtle inconsistencies could cause incorrect results.

---

## 1. Current Implementation Map

### 1.1 Unit Storage Architecture

Class Hierarchy:
- NDArray (basearrays/ndarray.py): _units attribute, units property with validation
- NDMath (arraymixins/ndmath.py): _op(), _resolve_operation_units() for arithmetic
- Coord (dataset/coord.py): Inherits from NDMath+NDArray, has units
- NDDataset (dataset/nddataset.py): Inherits from NDMath+NDIO+NDComplexArray, has units
- CoordSet (dataset/coordset.py): units property returns list of coord units

### 1.2 Unit Registry

Location: core/units/__init__.py
- Custom UnitRegistry with spectroscopy context
- Custom formatters (ScpDefaultFormatter, ScpCompactFormatter, etc.)
- Spectroscopy-specific units: transmittance, absorbance, Kubelka_Munk, ppm
- Custom dotted symbol preprocessing for "a.u.", "K.M."

### 1.3 Objects That Can Carry Units

| Object | Storage | Type | Access |
|--------|---------|------|--------|
| NDDataset | _units | pint.Unit or None | dataset.units |
| Coord | _units | pint.Unit or None | coord.units |
| NDArray | _units | pint.Unit or None | array.units |
| CoordSet | N/A | List[pint.Unit] | coordset.units |
| Quantity | .units | pint.Unit | quantity.units |

### 1.4 Unit Propagation

Arithmetic: __add__ etc. -> _binary_op() -> _op() -> _resolve_operation_units() -> Pint Quantity arithmetic
Reductions: @_from_numpy_method -> _op() -> same unit resolution
Concatenation: Check data unit compatibility, auto-convert, but NO coord unit checking
Conversions: to(), ito(), to_base_units() with special spectroscopy handling

### 1.5 Key Lists for Unit Handling

__compatible_units: add, sub, iadd, isub, maximum, minimum, fmin, fmax, lt, le, ge, gt
__remove_units: logical_not, isfinite, isinf, isnan, isnat, isneginf, isposinf, iscomplex, signbit, sign
__require_units: trigonometric functions require dimensionless or specific units

---

## 2. Test Coverage Inventory

### 2.1 Existing Tests
- test_core/test_units/test_units.py: ppm, dotted symbols, basic arithmetic, repr
- test_core/test_dataset/test_dataset_units.py: invalid units, sqrt propagation, absorbance<->transmittance

### 2.2 Uncovered Areas
- Coord unit propagation in arithmetic
- Mixed Coord units in operations
- Concatenation with different coord units
- Comparison operations unit behavior
- Power operations unit behavior
- Quantity <-> NDDataset interoperability

### 2.3 Coverage Assessment: LOW-MEDIUM

---

## 3. Inconsistent or Underspecified Behaviors

### 3.1 CRITICAL ISSUES

#### 3.1.1 Coordinate Units IGNORED in Arithmetic Operations
- Problem: _op() operates on obj.data only, coord units completely ignored
- Impact: ds1 + ds2 checks data units but ignores coord unit compatibility
- Severity: HIGH - Silent semantic errors possible
- Location: arraymixins/ndmath.py line 3111

#### 3.1.2 Silent Unit Drop in Comparison Operations
- Problem: Comparisons in __compatible_units, forces np.add, but Pint comparison returns dimensionless
- Impact: Comparison returns NDDataset with units=None instead of dimensionless
- Severity: MEDIUM - Misleading unit information
- Location: arraymixins/ndmath.py lines 464, 2995-2996

#### 3.1.3 Power Operation Unit Semantics Underspecified
- Problem: pow transformed to exp(b*log(a)), loses exponent unit context
- Impact: dataset ** dataset2 with units may produce incorrect results
- Severity: MEDIUM-HIGH - Potential for physical nonsense
- Location: arraymixins/ndmath.py lines 3194-3196

### 3.2 MEDIUM PRIORITY ISSUES

#### 3.2.1 Inconsistent Quantity vs NDDataset Behavior
- Operations between Quantity and NDDataset have inconsistent return types
- Severity: MEDIUM

#### 3.2.2 Concatenation Does Not Align Coordinate Units
- Only data units checked and aligned, coord units concatenated without conversion
- Severity: MEDIUM
- Location: processing/transformation/concatenate.py lines 98-114

#### 3.2.3 CoordSet.units Returns Plain List
- No validation, order may not match dims, no access by dimension name
- Severity: LOW

#### 3.2.4 Reductions Return Quantity Scalars
- Inconsistent return type based on keepdims parameter
- Severity: LOW

### 3.3 LOW PRIORITY ISSUES
- Formatting inconsistencies across output contexts
- is_units_compatible method behavior unclear
- Unit setting auto-conversion potentially surprising

### 3.4 Summary
| Severity | Count | Issues |
|----------|-------|--------|
| HIGH | 2 | Coord units ignored, Power semantics |
| MEDIUM | 4 | Comparison drop, Quantity/NDDataset, Concat coords, CoordSet API |
| LOW | 4 | Reduction types, Formatting, is_units_compatible, Auto-conversion |

---

## 4. Conservative v1 Unit Contract Proposal

### 4.1 Guiding Principles
1. Preserve existing behavior where correct
2. Make implicit behavior explicit
3. Fail fast on ambiguous/dangerous operations
4. Minimal breaking changes
5. Consistent semantics

### 4.2 Contractual Guarantees
- NDArray.units, Coord.units are pint.Unit or None
- Setting units validates dimensionality (unless force=True)
- Arithmetic operations follow Pint Quantity arithmetic for data units
- Reductions preserve units when keepdims=True, return Quantity scalars when keepdims=False
- to()/ito() perform unit conversions correctly
- Spectroscopy-specific conversions work (absorbance <-> transmittance)

### 4.3 Known Limitations (v1)
1. Coord units NOT validated during arithmetic (documented limitation)
2. Power with dimensional exponents has undefined behavior
3. Mixed Quantity/NDDataset operations may have type inconsistencies
4. Concatenation does NOT auto-convert coordinate values

### 4.4 Recommended Documentation
```
SpectroChemPy v1 Unit Contract:
- Data units follow Pint Quantity arithmetic
- Coord units are metadata only (not validated in ops) [LIMITATION]
- Conversions: to(), itoo(), with special spectroscopy support
- Power with dimensional exponents undefined [LIMITATION]
```

---

## 5. Recommendations

### 5.1 IMMEDIATE (P0 - Do Now)

#### P0.1: Add Coordinate Unit Validation for Concatenation
**File:** processing/transformation/concatenate.py
**Change:** Add coord unit checking and alignment in concatenate()
**Risk:** LOW **Value:** HIGH

#### P0.2: Fix Comparison Operations
**File:** arraymixins/ndmath.py
**Change:** Remove comparisons from __compatible_units, handle separately to return dimensionless
**Risk:** LOW **Value:** MEDIUM

#### P0.3: Add Power Operation Validation
**File:** arraymixins/ndmath.py
**Change:** Validate exponent is dimensionless in _check_order()
**Risk:** LOW **Value:** HIGH

#### P0.4: Add Critical Unit Tests
**New File:** tests/test_core/test_units/test_unit_propagation.py
**Test:** Coord units, concatenation coords, power validation, comparisons, mixed types
**Risk:** NONE **Value:** HIGH

### 5.2 MEDIUM-TERM (P1 - Next Release)

#### P1.1: Redesign Coordinate Unit Semantics
**Options:** A=Keep as metadata with validation, B=Physical semantics, C=Separate geometry types
**Recommendation:** A for v1, C for v2

#### P1.2: Unified Quantity/NDDataset Interoperability
**Action:** Create conversion utilities and document patterns

#### P1.3: Complete Unit Documentation
**Action:** Create docs/source/units.rst with full specification

### 5.3 LONG-TERM (P2 - Future)

#### P2.1: Full Physical Quantity Integration
**Vision:** All numerical data as Pint Quantities internally
**Status:** Prototype needed, evaluate performance

#### P2.2: Dimensional Coordinate System
**Vision:** Coordinates have physical dimensions, operations validate compatibility

---

## Appendix A: File Reference

| Component | Files | Key Lines |
|-----------|-------|-----------|
| Unit Registry | core/units/__init__.py | 236-270 |
| NDArray units | core/dataset/basearrays/ndarray.py | 251, 2207-2230 |
| CoordSet units | core/dataset/coordset.py | 548-552 |
| Arithmetic ops | core/dataset/arraymixins/ndmath.py | 3073-3084, 2920-3009 |
| Unit requirements | core/dataset/arraymixins/ndmath.py | 436-513 |
| Conversions | core/dataset/basearrays/ndarray.py | 1916-2086 |
| Concatenation | processing/transformation/concatenate.py | 98-163 |
| Reductions | core/dataset/arraymixins/ndmath.py | 1978, 2289, 2382 |
| Tests | test_core/test_units/test_units.py | All |
| Tests | test_core/test_dataset/test_dataset_units.py | All |

---

## Appendix B: Questions Answered

Q1: Where are units stored? A: _units attribute in NDArray base class
Q2: Which objects can carry units? A: NDDataset, Coord, NDArray, CoordSet (list)
Q3: Which operations preserve units? A: Most arithmetic via Pint, reductions with keepdims=True
Q4: Which operations recompute units? A: All arithmetic via _resolve_operation_units()
Q5: Which operations silently drop/inconsistent? A: Comparisons (drop), Coord units (ignored), Power (undefined)
Q6: Are Coord.units metadata, geometry, or physical? A: **Metadata only** - significant limitation
Q7: How does Quantity interact? A: Used internally, but no seamless interoperability
Q8: Are conversions explicit, implicit, or incomplete? A: Mostly explicit via to()/ito(), but implicit in setting .units and arithmetic
Q9: What behavior is tested? A: Basic creation, validation, sqrt, spectroscopy conversions
Q10: What should be contractual? A: See Section 4 - v1 Unit Contract

---

*Audit Complete*
