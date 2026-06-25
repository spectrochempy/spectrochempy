# Maintainer Documentation

Ce dossier contient la documentation destinée aux **mainteneurs** du projet
SpectroChemPy : procédures de publication, récupération après incident,
contrats RFC, et notes d'architecture durables.

> **⚠️ Ce n'est pas de la documentation utilisateur.**
> La documentation publique se trouve sur
> [spectrochempy.fr](https://www.spectrochempy.fr) et dans le dossier
> [`docs/`](../docs/).

## Structure

| Path | Role | Purpose |
|---|---|---|---|
| [`release-process.md`](release-process.md) | How | Procédure complète de publication |
| [`emergency-recovery.md`](emergency-recovery.md) | How | Incidents connus et récupération |
| [`rfcs/`](rfcs/) | **What** | RFC mainteneur et contrats de comportement |
| [`architecture/`](architecture/) | **How** | Notes d'architecture durables et cartes de risques |
| [`audits/`](audits/) | **Why** | Audits historiques promus (contexte, pas autorité) |
| [`roadmap/`](roadmap/) | **When** | Feuilles de route et roadmaps de migration |

## Documentation Structure

Utiliser ces couches documentaires dans cet ordre :

| Layer | Use for | Look here first |
|---|---|---|
| RFCs | Contrats mainteneur, positions, décisions acceptées, et contrats implémentés | [`rfcs/INDEX.md`](rfcs/INDEX.md) |
| Architecture Notes | Références d'architecture durables, cartes d'implémentation, et contexte technique maintenu | [`architecture/INDEX.md`](architecture/INDEX.md) |
| Promoted Audits | Audits historiques promus pour contexte durable (non autoritaires) | [`audits/INDEX.md`](audits/INDEX.md) |
| Local Audits | Notes de travail, historique d'implémentation, investigations, et campagnes locales non versionnées par défaut | notes locales sous `audit/`, lorsqu'elles existent |
| Roadmap | Priorités, sujets terminés, actifs, ou différés | [`roadmap/architecture-roadmap.md`](roadmap/architecture-roadmap.md) |

### Current Maintainer Entry Points

Après les campagnes récentes de normalisation des readers, de persistance
portable, et de consolidation des Result Objects, les points d'entrée
mainteneur les plus utiles sont généralement :

- [`roadmap/architecture-roadmap.md`](roadmap/architecture-roadmap.md) pour l'état
  global des campagnes et futurs candidats ;
- [`architecture/metadata-and-support-model.md`](architecture/metadata-and-support-model.md)
  pour le modèle runtime central ;
- [`architecture/reader-normalization-architecture.md`](architecture/reader-normalization-architecture.md)
  pour la normalisation des imports readers ;
- [`architecture/portable-persistence-model.md`](architecture/portable-persistence-model.md)
  pour la surface portable xarray / NetCDF effectivement implémentée ;
- [`architecture/result-object-contract-rfc.md`](architecture/result-object-contract-rfc.md)
  pour le contrat Result Object maintenant stabilisé.

### Important Distinction: tracked architecture docs vs local audits

Le nom de certains fichiers suivis sous `maintainers/architecture/` contient le
mot `audit`, mais cela ne veut pas dire qu'ils sont équivalents aux notes
locales sous `audit/`.

- les fichiers suivis dans `maintainers/architecture/` sont des références
  mainteneur versionnées, indexées, et conservées pour le long terme ;
- les fichiers sous `audit/` restent par défaut des notes de campagne,
  d'investigation, de migration, ou d'historique d'implémentation.

En pratique : commencer par `maintainers/`, puis consulter `audit/` seulement
si le contexte historique détaillé est réellement nécessaire.

## Local audit notes

Audit files whose names start with `~` or `~~` are working notes. They may be
local-only and should not be treated as authoritative maintainer
documentation.

Before closing a campaign, maintainers should check whether local audit notes
contain architectural decisions, compatibility constraints, known
limitations, or roadmap decisions that future maintainers will need.

If so, summarize those conclusions in tracked `maintainers/` documentation or
in a compact campaign summary before considering the campaign complete.

En pratique :

- commencer par ce fichier ;
- puis lire la roadmap ;
- puis consulter l'index RFC ou l'index architecture selon que la question est
  normative ou surtout technique ;
- pour les questions sur dimensions, coordonnées, `CoordSet`, métadonnées,
  labels, provenance, commencer par
  [`architecture/metadata-and-support-model.md`](architecture/metadata-and-support-model.md) ;
- pour les questions sur ce qui survit actuellement aux round-trips portables
  xarray / NetCDF, commencer par
  [`architecture/portable-persistence-model.md`](architecture/portable-persistence-model.md) ;
- pour les questions sur l'import et la normalisation des readers, commencer
  par
  [`architecture/reader-normalization-architecture.md`](architecture/reader-normalization-architecture.md) ;
- utiliser les audits locaux seulement pour retrouver le contexte de campagne
  et l'historique d'implémentation.

### Document Role Framework

Chaque document dans `maintainers/` répond à une question fondamentale :

| Role | Question | Type de document |
|---|---|---|
| **What** | Qu'est-ce qui est contractuel ? | RFC |
| **How** | Comment le système est-il organisé ? | Architecture note |
| **When** | Dans quel ordre les changements arrivent-ils ? | Roadmap |
| **Why** | Pourquoi cette décision a-t-elle été prise ? | Audit |

Cette distinction aide à placer un nouveau contenu au bon endroit : un auteur
qui hésite entre RFC et architecture note peut se demander s'il définit un
contrat (What) ou décrit une implémentation (How).

### Architecture-Document Lifecycle

Le cycle documentaire typique est :

```text
Local audit (working notes)
      │
      ▼
RFC (if needed — defines contracts)
      │
      ▼
Implementation
      │
      ▼
Architecture note (durable reference)
      │
      ▼
Promoted historical audit (long-term context)
```

- les **audits locaux** sont des notes de travail et des analyses de campagne ;
- les **RFCs** portent des contrats proposés, acceptés, ou implémentés ;
- l'**implémentation** transforme le contrat en comportement mergé ;
- les **notes d'architecture** deviennent la référence suivie lorsqu'un design
  se stabilise ;
- les **audits promus** conservent le contexte historique durable qui ne fait
  plus autorité.

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

1. [`roadmap/architecture-roadmap.md`](roadmap/architecture-roadmap.md)
2. [`rfcs/INDEX.md`](rfcs/INDEX.md)
3. [`architecture/INDEX.md`](architecture/INDEX.md)
4. [`rfcs/project-invariants-rfc.md`](rfcs/project-invariants-rfc.md)
5. [`rfcs/project-copy-semantics-rfc.md`](rfcs/project-copy-semantics-rfc.md)
6. [`architecture/portable-persistence-model.md`](architecture/portable-persistence-model.md)
7. [`architecture/coordset-storage-architecture.md`](architecture/coordset-storage-architecture.md)
8. [`architecture/metadata-and-support-model.md`](architecture/metadata-and-support-model.md)
9. [`architecture/reader-normalization-architecture.md`](architecture/reader-normalization-architecture.md)
10. [`architecture/display-architecture.md`](architecture/display-architecture.md)
11. [`architecture/result-object-contract-rfc.md`](architecture/result-object-contract-rfc.md)
12. [`architecture/result-object-migration-roadmap.md`](architecture/result-object-migration-roadmap.md)
13. [`architecture/mathematical-semantics-and-metadata-propagation.md`](architecture/mathematical-semantics-and-metadata-propagation.md)
14. [`architecture/array-class-responsibility.md`](architecture/array-class-responsibility.md)
15. [`rfcs/coordinate-arithmetic-semantics.md`](rfcs/coordinate-arithmetic-semantics.md)
16. [`rfcs/trusted-and-portable-persistence.md`](rfcs/trusted-and-portable-persistence.md)

## Key Documents

| Document | Description |
|---|---|
| [`rfcs/INDEX.md`](rfcs/INDEX.md) | Index des RFCs mainteneur, de leur statut, et de leur rôle |
| [`architecture/INDEX.md`](architecture/INDEX.md) | Point d'entrée organisé vers les notes d'architecture suivies |
| [`roadmap/architecture-roadmap.md`](roadmap/architecture-roadmap.md) | Feuille de route légère des sujets d'architecture récents, terminés ou différés |
| [`architecture/portable-persistence-model.md`](architecture/portable-persistence-model.md) | Référence d'architecture principale pour la surface de persistance portable xarray / NetCDF actuellement implémentée |
| [`architecture/metadata-and-support-model.md`](architecture/metadata-and-support-model.md) | Référence d'architecture principale pour dimensions, coordonnées, `CoordSet`, labels, métadonnées et provenance |
| [`architecture/reader-normalization-architecture.md`](architecture/reader-normalization-architecture.md) | Référence d'architecture principale pour la normalisation des readers |
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
