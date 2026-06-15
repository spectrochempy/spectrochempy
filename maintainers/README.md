# Documentation interne des mainteneurs

Ce dossier contient la documentation destinée aux **mainteneurs** du projet
SpectroChemPy. Elle décrit les procédures de publication, de déploiement et
de récupération après incident.

> **⚠️ Ce n'est pas de la documentation utilisateur.**
> La documentation publique se trouve sur
> [spectrochempy.fr](https://www.spectrochempy.fr) et dans le dossier
> [`docs/`](../docs/).

## Documents

| Document | Description |
|----------|-------------|
| [`release-process.md`](release-process.md) | Procédure complète de publication (core, plugins, vérifications) |
| [`emergency-recovery.md`](emergency-recovery.md) | Incidents connus et résolutions |
| [`architecture/README.md`](architecture/README.md) | Index des notes d'architecture mainteneur versionnées |
| [`rfcs/architecture-roadmap.md`](rfcs/architecture-roadmap.md) | Feuille de route légère des sujets d'architecture récents, terminés ou différés |
| [`rfcs/metadata-contract.md`](rfcs/metadata-contract.md) | RFC mainteneur définissant la sémantique normative des métadonnées `NDDataset` |
| [`rfcs/coordinate-arithmetic-semantics.md`](rfcs/coordinate-arithmetic-semantics.md) | RFC mainteneur décrivant la position actuelle sur la sémantique de l'arithmétique coordonnée |
| [`display-architecture.md`](display-architecture.md) | Architecture d'affichage finale — documentation mainteneur pour la couche d'affichage HTML sémantique |

## Workflows GitHub associés

Les procédures décrites ici s'appuient sur les workflows GitHub Actions du
dépôt (`.github/workflows/`). Consultez directement ces fichiers YAML pour
le détail technique des étapes automatisées.
