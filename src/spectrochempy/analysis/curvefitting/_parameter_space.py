# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Adapters between fitting definitions and bounded optimizer vectors.

This preparation layer preserves the canonical and legacy parameter layouts.
It does not parse scripts, evaluate profiles, or select an optimization method.
"""

from spectrochempy.analysis.curvefitting._modelspec import _FitModelSpec
from spectrochempy.analysis.curvefitting._parameter_transform import _to_external
from spectrochempy.analysis.curvefitting._parameter_transform import _to_internal


def _modelspec_parameter_count(spec):
    """Return the FitParameters-compatible total parameter count for a spec."""
    return len(spec.common_params) + sum(len(comp.params) for comp in spec.components)


def _restore_modelspec_external(spec, values, keys):
    """Restore external parameter values in a model spec from optimizer values."""
    key_index = {key: idx for idx, key in enumerate(keys)}
    for key, ps in spec._iter_varying():
        lob, upb = ps.bounds
        if ps.name in spec.expvars:
            restored = []
            for i in range(spec.expnumber):
                exp_key = f"{key}_exp{i}"
                if exp_key not in key_index:
                    break
                restored.append(_to_external(values[key_index[exp_key]], lob, upb))
            if restored:
                ps.value = restored
        else:
            if key not in key_index:
                continue
            ps.value = _to_external(values[key_index[key]], lob, upb)
    return spec


def _modelspec_internal_vector(spec):
    """Return internal optimizer values and keys for varying spec parameters."""
    values = []
    keys = []
    for key, ps in spec._iter_varying():
        lob, upb = ps.bounds
        if ps.name in spec.expvars:
            for i in range(spec.expnumber):
                values.append(_to_internal(ps.value[i], lob, upb))
                keys.append(f"{key}_exp{i}")
        else:
            values.append(_to_internal(ps.value, lob, upb))
            keys.append(key)
    return values, keys


def _internal_vector(fp):
    """Return optimizer values and keys in historical parameter order."""
    if isinstance(fp, _FitModelSpec):
        return _modelspec_internal_vector(fp)

    par = []
    keys = []

    for key in sorted(fp.keys()):
        if not fp.fixed[key]:
            keysp = key.split("_")[0]
            if keysp in fp.expvars:
                for i in range(fp.expnumber):
                    par.append(
                        _to_internal(
                            fp.data[key][i],
                            fp.lob[key],
                            fp.upb[key],
                        ),
                    )
                    keys.append(f"{key}_exp{i}")
            else:
                par.append(
                    _to_internal(
                        fp.data[key],
                        fp.lob[key],
                        fp.upb[key],
                    ),
                )
                keys.append(key)

    return par, keys


def _restore_external(fp, p, keys):
    """Restore physical values in place using the supplied optimizer keys."""
    if isinstance(fp, _FitModelSpec):
        return _restore_modelspec_external(fp, p, keys)

    for key in list(fp.keys()):
        keysp = key.split("_")
        if keysp[0] in fp.expvars:
            ps = []
            for i in range(fp.expnumber):
                ks = f"{key}_exp{i}"
                if ks not in keys:
                    break
                k = keys.index(ks)
                ps.append(p[k])
            if ps:
                fp.data[key] = _to_external(
                    ps,
                    fp.lob[key],
                    fp.upb[key],
                )
        else:
            if key not in keys:
                continue
            k = keys.index(key)
            fp.data[key] = _to_external(
                p[k],
                fp.lob[key],
                fp.upb[key],
            )
    return fp
