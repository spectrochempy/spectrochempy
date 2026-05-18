# Cantera Plugin — Audit de couplage

> Généré le 2026-05-17 — branche `cantera-coupling-audit`
> Dernière exécution : 2026-05-17 — 276 tests pass, 5 skip
> Objectif : cartographier le couplage avant extraction plugin.

---

## 1. Modules couplés à Cantera

### 1.1 Core SpectroChemPy (`src/spectrochempy/`) — État APRÈS exécution

| Module | Niveau | Lignes | Détails |
|---|---|---|---|
| `analysis/kinetic/kineticutilities.py` | **AUCUN** | 0 | ~~~760 lignes~~ supprimées. `PFR` délégué conditionnellement au plugin. |
| `utils/optional.py` | **AUCUN** | 0 | Entrée `cantera` retirée de `VERSIONS`. |
| `api/plugins/base.py` | **FAIBLE** | 1 ligne | Exemple `requires = ["cantera>=3.0"]` dans docstring — documentation uniquement. |
| `api/plugins/validation.py` | **FAIBLE** | 1 ligne | Commentaire `"cantera>=3.0"` comme exemple. |
| `plugins/registries.py` | **FAIBLE** | 1 ligne | Docstring mentionne `"thermodynamic_package"` — usage générique. |

### 1.2 Plugin Cantera (`plugins/spectrochempy-cantera/`) — inchangé

| Module | Niveau | Lignes | Détails |
|---|---|---|---|
| `src/spectrochempy_cantera/__init__.py` | **FORT** | 452 | `CanteraPlugin`, `equilibrium_composition`, `reactor_profile`, `flame_speed`, `thermo_properties`, `kinetic_sensitivity`, `spectral_coupling`, `read_cantera_mechanism`. Tous avec `import cantera as ct` différé. |
| `src/spectrochempy_cantera/_pfr.py` | **FORT** | 794 | `PFR` class **(source unique désormais)**, `_cantera_is_not_available`, `_ct_modify_rate`, `_ct_modify_surface_kinetics`. |
| `tests/test_cantera.py` | **FORT** | 244 | Tests plugin. |

---

## 2. Problème critique : code dupliqué — RÉSOLU

Le core `kineticutilities.py` et le plugin `_pfr.py` **ne sont plus dupliqués**.

**Ce qui a été fait :**
- Suppression du `PFR` class, `_cantera_is_not_available`, `_ct_modify_rate`, `_ct_modify_surface_kinetics`, `SCIPY_MINIMIZE_METHODS` du core
- `PFR` est maintenant importé conditionnellement depuis `spectrochempy_cantera._pfr` s'il est installé, sinon un stub avec `ImportError` explicite est fourni
- `_cantera_is_not_available()` déléguée au plugin
- `api_methods.py` inchangé (auto-généré) — le lazy import vers `kineticutilities` fonctionne toujours via la délégation

---

## 3. Dépendances implicites

### Core → Plugin (après refactoring)

Le `PFR` du plugin importe depuis `spectrochempy.core` :
- `Coord`, `NDDataset` — couplage fort au modèle de données
- `error_`, `info_` — logging
- `scipy.optimize` — solveurs (indépendant de cantera)

Le plugin `__init__.py` importe depuis le core :
- `PluginCapability`, `SpectroChemPyPlugin`, `CORE_PLUGIN_API_VERSION`
- `NDDataset` (type hint)

### Lazy loading

- `ActionMassKinetics` et `PFR` sont dans `lazyimport/api_methods.py` comme lazy
- `import spectrochempy` **NE déclenche PAS** `import cantera` (vérifié)
- Le plugin utilise `import cantera as ct` différé dans toutes les fonctions
- Le core `kineticutilities.py` **n'importe plus cantera du tout**

---

## 4. Découpage proposé — Exécuté

### Phase 1 — Suppression de la duplication dans le core ✅

| Action | Fichier | Statut |
|---|---|---|
| Remplacer le `PFR` du core par une délégation conditionnelle | `kineticutilities.py:909-925` | ✅ |
| Migrer `_cantera_is_not_available` vers le plugin (délégation) | `kineticutilities.py:912-924` | ✅ |
| Supprimer `ct = import_optional_dependency(...)` du module-level | `kineticutilities.py` (supprimé) | ✅ |
| Supprimer `_ct_modify_rate`, `_ct_modify_surface_kinetics` du core | `kineticutilities.py` (supprimé) | ✅ |
| Supprimer `SCIPY_MINIMIZE_METHODS` du core | `kineticutilities.py` (supprimé) | ✅ |
| Nettoyer les imports superflus | `kineticutilities.py` | ✅ |
| Adapter `test_cu()` — ne plus patcher `ku.ct` | `test_kineticutilities.py:24` | ✅ |

Le `PFR` est maintenant **importé conditionnellement** depuis le plugin :
```python
try:
    from spectrochempy_cantera._pfr import PFR
    from spectrochempy_cantera._pfr import _cantera_is_not_available
except ImportError:
    class PFR:
        def __init__(self, *args, **kwargs):
            raise ImportError("PFR requires the 'spectrochempy-cantera' plugin...")
    def _cantera_is_not_available():
        return True
```

### Phase 2 — Ce qui reste dans le core après exécution

- `ActionMassKinetics` — totalement indépendant de cantera
- `Quantity`, `Coord`, `NDDataset` — modèles de données de base
- Plugin API (`SpectroChemPyPlugin`, `PluginCapability`, etc.)
- `ExtensionRegistry` et hooks de simulation (`register_simulations`)

### Phase 2 — Ce qui est maintenant uniquement dans le plugin

- `PFR` class (entièrement) — source unique dans `_pfr.py`
- `_ct_modify_rate`, `_ct_modify_surface_kinetics`
- `_cantera_is_not_available`
- `equilibrium_composition`, `reactor_profile`, `flame_speed`
- `thermo_properties`, `kinetic_sensitivity`, `spectral_coupling`
- `read_cantera_mechanism`

### Phase 3 — Nettoyage ✅

| Action | Fichier | Statut |
|---|---|---|
| Retirer `cantera` de `VERSIONS` dans `optional.py` | `utils/optional.py:15-19` | ✅ |
| Mettre à jour `api_methods.py` | auto-généré, inchangé (délégation via `kineticutilities`) | ✅ |

---

## 5. Risques

| Risque | Impact | Probabilité | Atténuation |
|---|---|---|---|
| Duplication core/plugin | Élevé | Certaine | Phase 1 prioritaire |
| `PFR` utilisé par du code existant (`fit_to_gas_concentrations`) | Élevé | Certaine | Shim rétrocompatible dans le core |
| Rétrocompatibilité : `scp.PFR` ne doit pas disparaître | Moyen | Haute | Garder un re-export conditionnel dans le core |
| Cantera API changes (3.x → 4.x) | Moyen | Faible | Isoler les appels CTI derrière une couche fine |
| `test_kineticutilities.py` dépend du PFR core | Moyen | Haute | Adapter les tests après migration |

---

## 6. Ordre conseillé d'extraction

```
1. Supprimer la duplication PFR core ↔ plugin
   └── Remplacer PFR core par délégation conditionnelle (try: from spectrochempy_cantera import PFR)

2. Extraire les helpers cantera du core
   └── Déplacer _ct_modify_rate, _ct_modify_surface_kinetics, _cantera_is_not_available

3. Extraire les fonctions de simulation/analyse
   └── equilibrium_composition, reactor_profile, flame_speed, etc. (déjà fait dans le plugin)

4. Nettoyer optional.py et lazyimports
   └── Retirer cantera de VERSIONS, rediriger PFR lazy import

5. Nettoyer la docstring et commentaires du core
   └── Éviter les mentions de cantera dans le code core générique
```

---

## 7. Résumé

| Critère | Évaluation |
|---|---|
| Modules couplés (core) **après** | 0 fort + 4 faibles (docstrings) |
| Lignes cantera-dépendantes (core) **après** | **0** (supprimées du core) |
| Lignes cantera-dépendantes (plugin) | ~1246 |
| Duplication core/plugin | **NON** — résolue |
| Import cantera au démarrage | **NON** — aucun code cantera dans le core |
| Extraction réaliste ? | **OUI** — déjà réalisée |
| Difficulté estimée | **Faible** (l'essentiel est fait) |
| Rétrocompatibilité | Préservée — `scp.PFR` existe toujours avec délégation |

### Verdict

> La duplication du `PFR` a été supprimée. Le core ne contient plus aucun code cantera.
> Le système de délégation conditionnelle assure la rétrocompatibilité : `scp.PFR` fonctionne
> si le plugin est installé, et lève un `ImportError` clair sinon.
> Les 276 tests passent (0 régression).
