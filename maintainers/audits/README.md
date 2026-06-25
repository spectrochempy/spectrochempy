# Maintainer Audits

This directory contains **promoted historical audits** — tracked documents that
were originally local working notes under `audit/` and later promoted into the
maintained documentation set because they retain long-term value for context,
decision-space analysis, or migration history.

These files are **not the primary authority** for current behavior. They are
historical references. For current contracts, see [`../rfcs/`](../rfcs/). For
durable architecture notes, see [`../architecture/`](../architecture/).

## What belongs here

- Pre-migration analyses that explain why a design was chosen.
- Decision-space audits that map tradeoffs considered during a campaign.
- Baseline characterizations of behavior before a redesign.
- Risk maps that remain relevant even after mitigation.

## What does **not** belong here

- Local working notes still in progress — those stay in the repository-level
  `audit/` directory (untracked).
- Current architecture notes or RFCs — those live in `../architecture/` or
  `../rfcs/`.

## Convention

Seuls les audits promus (réécrits et commités) doivent se trouver ici.

Ne jamais copier directement une note de travail locale depuis `audit/`.
Un audit promu est **réécrit, pas déplacé**. La note locale originale peut
rester dans `audit/`, être supprimée, ou être réécrite et commitée sous un nom
distinct. Dans tous les cas, le fichier commité ici doit être un document
autonome et révisé, pas une copie brute d'une note de travail.

## Reading guidance

Consult these files when you need:
- historical context for why a design decision was made;
- a baseline comparison before and after a migration;
- a risk map that may still inform future work.

Do **not** consult these files as the current maintained contract. Always
prefer [`../rfcs/`](../rfcs/) or [`../architecture/`](../architecture/) for
authoritative reference.
