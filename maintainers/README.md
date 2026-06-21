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

## Documentation Structure

Utiliser ces couches documentaires dans cet ordre :

| Layer | Use for | Look here first |
|---|---|---|
| RFCs | Contrats mainteneur, positions, décisions acceptées, et contrats implémentés | [`rfcs/INDEX.md`](rfcs/INDEX.md) |
| Architecture Notes | Références d'architecture durables, cartes d'implémentation, et contexte technique maintenu | [`architecture/INDEX.md`](architecture/INDEX.md) |
| Audits | Notes de travail, historique d'implémentation, investigations, et campagnes locales non versionnées par défaut | `audit/~*.md` |
| Roadmap | Priorités, sujets terminés, actifs, ou différés | [`rfcs/architecture-roadmap.md`](rfcs/architecture-roadmap.md) |

En pratique :

- commencer par ce fichier ;
- puis lire la roadmap ;
- puis consulter l'index RFC ou l'index architecture selon que la question est
  normative ou surtout technique ;
- utiliser les audits locaux seulement pour retrouver le contexte de campagne
  et l'historique d'implémentation.

### Architecture-Document Lifecycle

Le cycle documentaire typique est :

```text
Audit
  ↓
RFC
  ↓
Architecture Note
```

- les audits sont des analyses de travail et des notes de campagne ;
- les RFCs portent des contrats proposés, acceptés, ou implémentés ;
- les notes d'architecture deviennent la référence suivie lorsqu'un design ou
  une campagne s'est stabilisé.

### Authority Guide

Considérer comme **authoritative** en priorité :

- la roadmap mainteneur ;
- les notes d'architecture suivies ;
- les RFCs implémentés ;
- les contrats et positions acceptés.

Considérer comme **non-authoritative by default** :

- les audits ;
- les notes d'implémentation ;
- les journaux de campagne ;
- les rapports de caractérisation.

En cas de doute, commencer par la roadmap, puis l'index RFC, puis l'index
architecture.

## Core Architecture Reading Path

Pour un nouveau mainteneur, l'ordre de lecture recommandé est :

1. [`rfcs/architecture-roadmap.md`](rfcs/architecture-roadmap.md)
2. [`rfcs/INDEX.md`](rfcs/INDEX.md)
3. [`architecture/INDEX.md`](architecture/INDEX.md)
4. [`rfcs/project-invariants-rfc.md`](rfcs/project-invariants-rfc.md)
5. [`rfcs/project-copy-semantics-rfc.md`](rfcs/project-copy-semantics-rfc.md)
6. [`architecture/coordset-storage-architecture.md`](architecture/coordset-storage-architecture.md)
7. [`architecture/display-architecture.md`](architecture/display-architecture.md)
8. [`architecture/result-object-contract-rfc.md`](architecture/result-object-contract-rfc.md)
9. [`architecture/result-object-migration-roadmap.md`](architecture/result-object-migration-roadmap.md)
10. [`rfcs/metadata-contract.md`](rfcs/metadata-contract.md)
11. [`architecture/mathematical-semantics-and-metadata-propagation.md`](architecture/mathematical-semantics-and-metadata-propagation.md)
12. [`architecture/array-class-responsibility.md`](architecture/array-class-responsibility.md)
13. [`rfcs/coordinate-arithmetic-semantics.md`](rfcs/coordinate-arithmetic-semantics.md)
14. [`rfcs/trusted-and-portable-persistence.md`](rfcs/trusted-and-portable-persistence.md)
15. [`rfcs/nddataset-xarray-mapping-specification.md`](rfcs/nddataset-xarray-mapping-specification.md)

## Key Documents

| Document | Description |
|---|---|
| [`rfcs/INDEX.md`](rfcs/INDEX.md) | Index des RFCs mainteneur, de leur statut, et de leur rôle |
| [`architecture/INDEX.md`](architecture/INDEX.md) | Point d'entrée organisé vers les notes d'architecture suivies |
| [`rfcs/architecture-roadmap.md`](rfcs/architecture-roadmap.md) | Feuille de route légère des sujets d'architecture récents, terminés ou différés |
| [`rfcs/project-invariants-rfc.md`](rfcs/project-invariants-rfc.md) | Invariants « Project » : ownership, cycle, doublons, identité clé/nom (implémenté) |
| [`rfcs/project-copy-semantics-rfc.md`](rfcs/project-copy-semantics-rfc.md) | Contrat de copie `Project` et sémantique deep/shallow maintenue |
| [`rfcs/metadata-contract.md`](rfcs/metadata-contract.md) | Contrat mainteneur pour les métadonnées `NDDataset` |
| [`rfcs/analysis-fit-result-architecture.md`](rfcs/analysis-fit-result-architecture.md) | RFC draft sur l'architecture actuelle des résultats d'analyse et de fit |
| [`rfcs/modeldata-semantic-contract.md`](rfcs/modeldata-semantic-contract.md) | Audit et décision sur la suppression de `NDDataset.modeldata` |
| [`rfcs/roi-semantic-contract.md`](rfcs/roi-semantic-contract.md) | Audit et décision sur la suppression de `NDDataset.roi` |
| [`rfcs/coordinate-arithmetic-semantics.md`](rfcs/coordinate-arithmetic-semantics.md) | Position actuelle sur la sémantique de l'arithmétique coordonnée |
| [`architecture/README.md`](architecture/README.md) | Index consolidé des notes d'architecture |
| [`architecture/array-class-responsibility.md`](architecture/array-class-responsibility.md) | Responsabilités actuelles des classes array |
| [`architecture/mathematical-semantics-and-metadata-propagation.md`](architecture/mathematical-semantics-and-metadata-propagation.md) | RFC de caractérisation largement stabilisé sur les opérations mathématiques, result assembly, identité, provenance et métadonnées |
| [`architecture/display-architecture.md`](architecture/display-architecture.md) | Architecture d'affichage finale |

## Workflows GitHub associés

Les procédures décrites ici s'appuient sur les workflows GitHub Actions du
dépôt (`.github/workflows/`). Consultez directement ces fichiers YAML pour
le détail technique des étapes automatisées.
