# Maintainer Documentation

Ce dossier contient la documentation destinée aux **mainteneurs** du projet
SpectroChemPy : procédures de publication, récupération après incident,
contrats RFC, et notes d'architecture durables.

> **⚠️ Ce n'est pas de la documentation utilisateur.**
> La documentation publique se trouve sur
> [spectrochempy.fr](https://www.spectrochempy.fr) et dans le dossier
> [`docs/`](../docs/).

## Structure

| Path | Purpose |
|---|---|
| [`release-process.md`](release-process.md) | Procédure complète de publication |
| [`emergency-recovery.md`](emergency-recovery.md) | Incidents connus et récupération |
| [`rfcs/`](rfcs/) | RFC mainteneur et contrats de comportement |
| [`architecture/`](architecture/) | Audits, notes d'architecture, et cartes de risques durables |

## Key Documents

| Document | Description |
|---|---|
| [`rfcs/architecture-roadmap.md`](rfcs/architecture-roadmap.md) | Feuille de route légère des sujets d'architecture récents, terminés ou différés |
| [`rfcs/metadata-contract.md`](rfcs/metadata-contract.md) | Contrat mainteneur pour les métadonnées `NDDataset` |
| [`rfcs/modeldata-semantic-contract.md`](rfcs/modeldata-semantic-contract.md) | Audit et décision sur la suppression de `NDDataset.modeldata` |
| [`rfcs/roi-semantic-contract.md`](rfcs/roi-semantic-contract.md) | Audit et décision sur la suppression de `NDDataset.roi` |
| [`rfcs/coordinate-arithmetic-semantics.md`](rfcs/coordinate-arithmetic-semantics.md) | Position actuelle sur la sémantique de l'arithmétique coordonnée |
| [`architecture/README.md`](architecture/README.md) | Index consolidé des notes d'architecture |
| [`architecture/array-class-responsibility.md`](architecture/array-class-responsibility.md) | Responsabilités actuelles des classes array |
| [`architecture/mathematical-semantics-and-metadata-propagation.md`](architecture/mathematical-semantics-and-metadata-propagation.md) | Audit/RFC en cours sur les opérations mathématiques, result assembly, identité, provenance et métadonnées |
| [`architecture/display-architecture.md`](architecture/display-architecture.md) | Architecture d'affichage finale |

## Workflows GitHub associés

Les procédures décrites ici s'appuient sur les workflows GitHub Actions du
dépôt (`.github/workflows/`). Consultez directement ces fichiers YAML pour
le détail technique des étapes automatisées.
