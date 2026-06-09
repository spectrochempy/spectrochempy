# Procédure de publication

## Prérequis

### Secrets GitHub nécessaires

Les secrets suivants doivent être définis dans
[Settings → Secrets and variables → Actions](https://github.com/spectrochempy/spectrochempy/settings/secrets/actions)
du dépôt `spectrochempy/spectrochempy` :

| Secret | Usage |
|--------|-------|
| `PYPI_API_TOKEN` | Publication **plugins** sur PyPI (via `pypa/gh-action-pypi-publish` avec `password`) |
| `TEST_PYPI_API_TOKEN` | Publication **plugins** sur Test PyPI (workflow `publish_plugins.yml`) |
| `ANACONDA_API_TOKEN` | Publication sur Anaconda.org (compte `spectrocat`) — core + plugins |
| `BOT_TOKEN` | PAT personnel utilisé pour contourner la protection de branche lors des releases de plugins (expire tous les 3 mois — penser à le renouveler et mettre à jour le secret) |

> **Note PyPI core** : le package **core** (`spectrochempy`) utilise
> [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC)
> via le workflow `build_package.yml`.  Il n'utilise **pas**
> `PYPI_API_TOKEN` ni `TEST_PYPI_API_TOKEN`.  Les secrets API token ne
> sont requis que pour la publication des **plugins**
> (`publish_plugins.yml`).
>
> Avant la première release core via Trusted Publishing, vérifier que
> le workflow `build_package.yml` est bien configuré comme Trusted
> Publisher dans les paramètres PyPI et TestPyPI du projet
> `spectrochempy`.

### Comptes externes

- **Zenodo (core uniquement)** : l'intégration GitHub → Zenodo doit être activée sur le dépôt
  ([instructions Zenodo](https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content)).
  Une Release GitHub du **core** (tag `spectrochempy-vX.Y.Z`) déclenche automatiquement l'archivage DOI.
  Les releases plugins (tag `spectrochempy-XXX-vX.Y.Z`) ne doivent **pas** être archivées par Zenodo
  (voir [Zenodo and plugin releases](#zenodo-and-plugin-releases)).

## Vérifications préalables des services externes

Avant de lancer une release (core ou plugin), vérifier l'état des
services externes.

### Zenodo

**Avant une release du core :**

1. Aller sur la [page GitHub de Zenodo](https://zenodo.org/account/settings/github/)
2. Ouvrir l'onglet **GitHub** (premier onglet, par défaut)
3. Chercher `spectrochempy/spectrochempy` dans la liste des dépôts
4. Vérifier que le bouton à droite indique **On** (vert) — pas **Off** (grisé)
5. Si le dépôt est grisé (Off), cliquer sur le bouton pour le réactiver (On)
6. Si le dépôt est déjà On mais que l'intégration semble ne pas
   fonctionner (par exemple après une phase de releases plugins), on peut
   **toggle** (Off → On) pour forcer Zenodo à reconnaître le dépôt
7. Vérifier l'onglet **Errors** (deuxième onglet) : aucune erreur active
   (pas de croix rouge)
8. Vérifier que `CITATION.cff` et `zenodo.json` sont valides (les versions
   sont correctes)

**Avant une release de plugins :**

- Vérifier que l'intégration GitHub est **Off** (grisée) dans Zenodo
  (voir [Zenodo and plugin releases](#zenodo-and-plugin-releases))

### Anaconda.org

- L'organisation `spectrocat` contient le package attendu :
  ```bash
  anaconda show spectrocat/spectrochempy
  ```
  ```bash
  anaconda show spectrocat/spectrochempy-nmr
  anaconda show spectrocat/spectrochempy-iris
  anaconda show spectrocat/spectrochempy-tensor
  anaconda show spectrocat/spectrochempy-hypercomplex
  anaconda show spectrocat/spectrochempy-carroucell
  ```

- **Première release d'un plugin** : si le package n'existe pas encore sur
  Anaconda, vérifier que `ANACONDA_API_TOKEN` a les droits de création de
  nouveaux packages sur l'organisation `spectrocat`.

  → Si la création automatique échoue, créer le package manuellement avec le
  fichier `.conda` construit par CI, puis relancer la release :
  `anaconda upload -l main <fichier>.conda`.

  Le `ANACONDA_API_TOKEN` utilisé par le workflow doit avoir les droits
  de **création** de nouveaux packages sur l'organisation `spectrocat`.

---

## Release du core

> **Note Zenodo** : avant une release du core, vérifier que l'intégration
> GitHub est active dans Zenodo. Si elle a été désactivée pour une phase
> de release plugin, la réactiver (voir
> [Zenodo and plugin releases](#zenodo-and-plugin-releases)).

### 1. Vérifier l'état de `master`

```bash
git fetch upstream
git switch master
git merge --ff-only upstream/master
git status
```

La branche doit être propre et à jour. Les checks CI doivent tous passer.

### 2. Lancer le workflow

Depuis l'interface GitHub :

1. Aller dans **Actions** → **Prepare a new release**
2. Cliquer **Run workflow**
3. Renseigner les paramètres :

```
versionString = X.Y.Z                          # (ex: 0.9.0 ou 0.10.0)
confirm_zenodo_enabled = true                  # ← cocher après avoir vérifié Zenodo
```

> **Important** : avant de lancer le workflow, vérifier que l'intégration
> GitHub → Zenodo est **On** (voir [Zenodo](#zenodo) ci-dessus).
> La case `confirm_zenodo_enabled` doit être cochée pour que le workflow
> démarre — cela garantit que Zenodo est prêt à archiver la future release
> sans intervention manuelle au moment de la publication.

### 3. Déroulement automatisé

Le workflow :

- Crée une branche `release/X.Y.Z`
- Met à jour les fichiers suivants :
  - `docs/sources/whatsnew/latest.rst`
  - `CITATION.cff`
  - `zenodo.json`
- Ouvre une **Pull Request** vers `master`

### 4. Vérifier la PR de release

Dans la Pull Request :

- Vérifier les fichiers modifiés (release notes, CITATION.cff, zenodo.json)
- Le titre doit être : `Release version X.Y.Z`
- Vérifier que les versions sont correctes
- Si des dépendances ont changé, vérifier `pyproject.toml` et
  `environments/environment_build.yml` également

### 5. Merge de la PR

- Cliquer **Merge pull request**
- La fusion déclenche automatiquement le workflow
  **Publish a draft new release**

### 6. Draft Release GitHub

Le workflow `publish_draft_new_release.yml` crée une **Draft Release** avec :

- Tag : `spectrochempy-vX.Y.Z`
- Titre : `SpectroChemPy vX.Y.Z`

Aller sur la
[page des releases](https://github.com/spectrochempy/spectrochempy/releases)
pour vérifier la Draft.

### 7. Publier la Release GitHub

- Éditer la Draft Release si nécessaire (ajouter des notes)
- Cliquer **Publish release**
- La publication déclenche automatiquement le workflow
  **Build and publish packages** qui publie sur :

  - **PyPI** (label stable, sans `--force`)
  - **Anaconda.org** (label `main`, avec `--force` pour déplacer une build
    déjà publiée sur `dev` vers le label stable)
  - **Zenodo** (via l'intégration GitHub)

- La publication déclenche également le workflow **Docs** (`build_docs.yml`)
  via l'événement `release: [published]` :

  > **Note sur la construction de la documentation** : ce build est volontaire.
  > Il vérifie que la documentation de release peut être construite avec le
  > tag publié (`spectrochempy-vX.Y.Z`). Il alimente aussi la documentation
  > versionnée (accessible sous `/<version>/`) et le dropdown des versions.
  > **Ne pas supprimer ce job** dans le workflow `build_docs.yml`.

### Modèle actuel de documentation versionnée

Le site publié par GitHub Pages est construit dans la branche `gh-pages` :

- la documentation `latest` est publiée à la racine du site et correspond à
  l'état courant de `master` ;
- chaque release stable du core est publiée dans un répertoire `X.Y.Z/`
  (par exemple `0.9.2/`) ;
- le dropdown des versions est généré à partir des répertoires semver présents
  dans `gh-pages` ;
- les tags Git du core utilisent le format canonique `spectrochempy-vX.Y.Z`,
  mais le répertoire public de documentation reste `X.Y.Z/` ;
- les tags plugins (`spectrochempy-<plugin>-vX.Y.Z`) ne doivent pas créer de
  documentation stable séparée.

`latest.rst` ne doit pas être modifié manuellement : il est régénéré depuis
`docs/sources/whatsnew/changelog.rst` par le hook pre-commit.

---

## Vérifications post-release

Après publication, vérifier que tout est accessible :

```bash
# PyPI
pip index versions spectrochempy
pip install spectrochempy==X.Y.Z

# Anaconda
anaconda show spectrocat/spectrochempy

# Version installée
python -c "import spectrochempy; print(spectrochempy.__version__)"
```

Vérifier également que le DOI Zenodo a été mis à jour sur la
[page Zenodo](https://zenodo.org/communities/spectrochempy).

Vérifier enfin la documentation :

- le workflow **Docs** (`build_docs.yml`) a réussi après publication de la
  release ;
- `https://www.spectrochempy.fr/X.Y.Z/` existe pour la nouvelle version ;
- le dropdown des versions contient `X.Y.Z` ;
- la racine du site affiche toujours la documentation `latest`.

Si la version n'apparaît pas dans le dropdown alors que la release existe :

1. Vérifier que `gh-pages` contient bien le répertoire `X.Y.Z/`.
2. Si le répertoire existe, lancer **Actions → Repair docs version index** :
   ce workflow régénère seulement `versions.json` et le dropdown, sans
   reconstruire toute la documentation.
3. Si le répertoire est absent, relancer manuellement
   **Actions → Docs → Run workflow** depuis `master`.
4. Vérifier que le tag core suit le format `spectrochempy-vX.Y.Z`.
5. Ne pas créer de tag alias local `X.Y.Z` : `docs/make.py -T` accepte les
   tags canoniques `spectrochempy-vX.Y.Z`.

---

## plugin_version_status.py — le moteur de décision central

Le script `.github/workflows/scripts/plugin_version_status.py` est le
composant central qui détermine l'état de publication des plugins.

### Rôle

- Trouve le dernier tag de release d'un plugin (`spectrochempy-XXX-v*`)
- Détecte les changements depuis ce tag dans les fichiers distribués
  (`src/`, `pyproject.toml`, `recipe.yaml`, `meta.yaml`, `README.md`,
  `LICENSE`, `MANIFEST.in`)
- Compte le nombre de commits pertinents depuis le dernier tag
- Calcule une version de développement (`next_patch.devN`) selon PEP 440
- Détermine si le plugin doit être publié (`has_changes`)

### Script unique, trois workflows

Un seul script est utilisé par trois workflows, ce qui garantit que la
logique de détection des changements reste cohérente :

| Workflow | Usage du script |
|----------|----------------|
| `plugin_release_status.yml` | `--all-official --summary` — Affiche un tableau read-only dans le résumé du workflow |
| `release_plugin.yml` | `--all-official --summary` — Affiche le même tableau comme confirmation avant release |
| `publish_plugins.yml` | `--plugin <name> --apply-dev-version --github-output --json` — Injecte la version de développement et expose les métadonnées |

### Pourquoi pas de logique dupliquée dans les workflows

Les workflows YAML n'ont pas besoin de recalculer les versions ou de
détecter les changements : ils déléguent tout au script Python. Cela évite
les divergences entre la logique de statut (read-only) et la logique de
build/publication.

### Première release d'un plugin

Quand un plugin n'a encore aucun tag de release :

- `plugin_version_status.py` retourne `latest_tag: ""` et
  `has_changes: true` (car tous les fichiers du plugin sont considérés
  comme nouveaux)
- La version de base est lue depuis `pyproject.toml`
- La version de développement est calculée comme `X.Y.Z.dev<N>` (où
  `X.Y.Z` est la version du fichier et `<N>` le nombre de commits
  depuis le début)
- Le tableau de statut affiche :no_entry: `no previous plugin tag`

### Tests dédiés

Le script a des tests unitaires dans
`tests/test_core/test_scripts/test_plugin_version_status.py` qui
couvrent :

- `test_parse_plugin_tag` — parsing des tags plugins spécifiques
- `test_next_patch_dev_version_is_newer_than_latest_stable` — calcul de
  `next_patch.devN`
- `test_release_relevant_paths_excludes_tests_and_docs` — filtrage des
  chemins pertinents
- `test_apply_dev_version_updates_plugin_metadata` — application de la
  version de développement

---

## Décider si un plugin nécessite une release

Avant de publier un plugin, comparer les changements depuis son dernier
tag publié.

### Workflow automatique (recommandé)

Le workflow **Check plugin release status** (`plugin_release_status.yml`)
s'exécute automatiquement à chaque push sur `master` et peut aussi être
déclenché manuellement depuis Actions → **Check plugin release status** →
**Run workflow**.

Il utilise `plugin_version_status.py --all-official --summary` pour
produire un tableau de synthèse dans le *workflow summary* (onglet
Summary du run) listant les plugins officiels avec :

- Statut (unchanged / modified / no previous tag)
- Dernier tag publié
- Nombre de commits ayant touché les fichiers distribués modifiés du plugin
- Nombre de fichiers distribués modifiés
- Version de développement calculée pour les builds non-release

Ce tableau permet de décider en un coup d'œil si un plugin mérite une
nouvelle release.

> **Attention** : le workflow ne vérifie pas la *nature* des changements
> (un changement de métadonnées distribuées comme `pyproject.toml` ou
> `recipe.yaml` peut suffire à marquer le plugin comme modifié). Utiliser la
> commande `git log` ci-dessous pour inspecter le détail si nécessaire.

### Trouver le dernier tag

```bash
git tag --list 'spectrochempy-XXX-v*' --sort=-v:refname
git log --oneline spectrochempy-XXX-v0.1.1..HEAD -- plugins/spectrochempy-XXX
```

### Vérifier la dernière version publiée

```bash
pip index versions spectrochempy-XXX
anaconda show spectrocat/spectrochempy-XXX
```

### Qu'est-ce qui justifie une nouvelle release ?

Un plugin mérite une nouvelle release si **des fichiers distribués** ont
changé depuis le dernier tag :

- `src/` (code livré aux utilisateurs)
- `pyproject.toml` (métadonnées, dépendances, entry points)
- `recipe.yaml` (recette conda)
- Fichiers inclus dans le package via `include` / `MANIFEST.in`
- Dépendances ajoutées, supprimées ou modifiées
- Compatibilité avec la nouvelle version du core
- Bug runtime corrigé

### Qu'est-ce qui ne justifie PAS une release ?

Un changement limité à l'un des éléments suivants ne nécessite
généralement pas de publication :

- Tests uniquement
- Documentation interne au dépôt
- CI / workflows GitHub
- Refactoring sans impact utilisateur

### Numérotation des versions

- **Ne jamais réutiliser** une version déjà publiée sur PyPI ou conda.
- Si `0.1.1` existe déjà et que le plugin a changé, publier `0.1.2`.
- Les builds de développement calculent automatiquement `next_patch.devN`
  depuis le dernier tag plugin et le nombre de commits ayant touché les
  fichiers distribués qui diffèrent encore du tag. Exemple : après
  `spectrochempy-XXX-v0.1.3`, avec 12 commits pertinents, la version de build
  est `0.1.4.dev12`.
- `next_patch.devN` est volontaire : selon PEP 440, `0.1.3.dev12` serait plus
  ancien que `0.1.3`, alors que `0.1.4.dev12` est bien plus récent que la
  dernière stable.
- Avant de choisir une version, vérifier :
  - [PyPI](https://pypi.org/project/spectrochempy-XXX/#history)
  - Anaconda : `anaconda show spectrocat/spectrochempy-XXX`
  - Tags GitHub : `git tag --list 'spectrochempy-XXX-v*'`

---

## Release des plugins

> **Important Zenodo** : avant de publier des plugins, désactiver
> l'intégration GitHub dans Zenodo (voir
> [Zenodo and plugin releases](#zenodo-and-plugin-releases)).
> La réactiver uniquement pour la prochaine release du core.

### Workflow

Si nécessaire, lancer d'abord **Actions** → **Check plugin release status**
pour vérifier quels plugins ont changé depuis leur dernier tag publié.

Ensuite, depuis **Actions** → **Release an official plugin**, exécuter le
workflow depuis la branche `master` avec les paramètres :

```
plugin_name: spectrochempy-XXX
version: X.Y.Z
confirm_zenodo_disabled: true   # ← doit être coché
```

> Le workflow refuse de démarrer si `confirm_zenodo_disabled` n'est pas coché.
> Cela garantit que l'intégration Zenodo a été désactivée avant la publication.

> **Note** : avant les étapes de bump, le workflow affiche un tableau
> dans le *step summary* listant tous les plugins officiels et leur
> statut (modifié depuis le dernier tag, inchangé, ou sans tag
> préexistant). Ce tableau est purement informatif : il aide le
> mainteneur à identifier quels autres plugins pourraient nécessiter
> une release, sans bloquer ni modifier la publication en cours.

### Déroulement

1. Le workflow **Release an official plugin** (`release_plugin.yml`) :
   - Vérifie que le workflow est déclenché depuis `master`
   - Vérifie que le plugin est dans la liste officielle
   - Bump la version dans `pyproject.toml`, `recipe.yaml` et `__init__.py`
   - Si la version était déjà à jour (cas de la **première release** d'un
     plugin, où la version a déjà été commitée sur `master`), le workflow
     détecte qu'aucun changement n'est nécessaire et **saute les étapes
     de commit et de push** — le tag et la Release sont créés depuis le
     HEAD existant
   - Pousse le commit sur `master` (via `BOT_TOKEN`) uniquement si un
     bump de version a eu lieu
   - Crée le tag `spectrochempy-XXX-vX.Y.Z`
   - Crée une Release GitHub
   - **Désactive le flag "Latest"** de la Release plugin, afin que la
     page d'accueil du dépôt continue d'afficher la dernière release du
     **core** comme "Latest" (cf. [GitHub "Latest" release handling](#github-latest-release-handling))
2. La Release GitHub déclenche automatiquement :
   - `publish_plugins.yml` → publication **PyPI**
   - `build_package.yml` → publication **Anaconda.org** (label `main`)

### Cas particulier : première release d'un plugin

Lors de la première release d'un plugin :

- La version du plugin est généralement déjà présente dans
  `pyproject.toml`, `recipe.yaml` et `__init__.py` depuis le commit
  d'ajout du plugin
- Le workflow détecte que toutes les sources de version sont déjà à jour
  et **saute l'étape de commit** (`git diff --cached --quiet`)
- Les étapes de rebase et de push sont également sautées
- Le tag et la Release GitHub sont créés normalement

Ce comportement est intentionnel : il évite de créer un commit vide de
bump de version quand la version cible est déjà en place.

### GitHub "Latest" release handling

Parce que le core et les plugins partagent le même dépôt GitHub, une
release plugin peut être automatiquement marquée comme **Latest** par
GitHub, prenant la place de la dernière release du core sur la page
d'accueil du dépôt.

Le workflow `release_plugin.yml` désactive automatiquement le flag
"Latest" après chaque création de release plugin (via
`gh release edit <tag> --latest=false`).

La release du core reste donc toujours celle affichée comme **Latest**
dans l'interface GitHub. Ce comportement est automatique et ne nécessite
aucune intervention manuelle.

### Vérification

```bash
pip install spectrochempy-XXX==X.Y.Z
anaconda show spectrocat/spectrochempy-XXX
```

---

## Zenodo and plugin releases

### Contexte

SpectroChemPy est un monorepo contenant le core et plusieurs plugins
officiels. Zenodo est connecté au dépôt GitHub
`spectrochempy/spectrochempy` et archive automatiquement toutes les
GitHub Releases si l'intégration est active.

### Problème

Lors de la publication des plugins, Zenodo crée des entrées pour les tags
plugin (`spectrochempy-iris-v0.1.1`, `spectrochempy-nmr-v0.1.1`, …). Ces
entrées sont incorrectes car :

- Elles archivent le dépôt monorepo complet avec les métadonnées globales
  de SpectroChemPy (titre, description, auteurs)
- Elles créent des DOI pour des releases qui ne représentent pas des
  versions du core
- Le titre Zenodo affiche "SpectroChemPy…" mais avec la version du plugin

### Politique

- **Zenodo doit être réservé aux releases du core SpectroChemPy** (tags
  `spectrochempy-vX.Y.Z`).
- **Les releases plugins ne doivent pas être archivées dans Zenodo** tant
  que les plugins restent dans le monorepo.
- Si un plugin nécessite son propre DOI à long terme, il devra soit être
  déplacé dans un dépôt séparé, soit utiliser une procédure Zenodo
  manuelle/spécifique à ce plugin. Sinon, les plugins ne doivent pas créer
  d'entrées Zenodo séparées.

### Procédure opérationnelle

1. **Avant de publier des plugins**, désactiver temporairement
   l'intégration GitHub du dépôt `spectrochempy/spectrochempy` dans Zenodo
   :
   - Aller sur [zenodo.org → GitHub](https://zenodo.org/account/settings/github/)
   - Chercher le dépôt `spectrochempy/spectrochempy` dans la liste
   - Vérifier que le bouton à droite indique **Off** (grisé) ; ou passer
     sur **Off** s'il indique **On**
   - Vérifier que la croix rouge est absente (l'état grisé signifie
     désactivé, pas en erreur)
2. **Publier les plugins** via le workflow **Release an official plugin**
   - Le workflow demande de cocher `confirm_zenodo_disabled` — le faire
     uniquement après avoir désactivé Zenodo
   - Si la case n'est pas cochée, le workflow échoue immédiatement avec
     un message explicite
3. **Vérifier PyPI et Anaconda.org** :
   ```bash
   pip install spectrochempy-XXX==X.Y.Z
   anaconda show spectrocat/spectrochempy-XXX
   ```
4. **Ne réactiver Zenodo** que pour la release du core suivante :
   - Aller sur [zenodo.org → GitHub](https://zenodo.org/account/settings/github/)
   - Chercher le dépôt `spectrochempy/spectrochempy`
   - Vérifier que le bouton à droite indique **On** (vert) ; ou passer
     sur **On** s'il indique **Off**
   - Vérifier que la croix rouge est absente

> **Rappel** : l'état Zenodo doit toujours être **On** pendant une
> release du core et **Off** pendant une release de plugins.
> Ne jamais laisser Zenodo actif pendant une release plugin.

---

## Ordre recommandé

1. **Release du core** → attendre la fin des builds CI
2. **Vérifier PyPI** : `pip install spectrochempy==X.Y.Z`
3. **Vérifier Anaconda** : `anaconda show spectrocat/spectrochempy`
4. **Vérifier Zenodo** : le DOI doit pointer vers la nouvelle version du
   core
5. **Désactiver Zenodo** (voir
   [Zenodo and plugin releases](#zenodo-and-plugin-releases))
6. **Release des plugins** (dans cet ordre) :
   - `spectrochempy-nmr`
   - `spectrochempy-iris`
   - `spectrochempy-tensor`
   - `spectrochempy-hypercomplex`
   - `spectrochempy-carroucell`

> **Note** : `spectrochempy-cantera` est actuellement **expérimental**
> et n'est pas publié automatiquement par les workflows CI. Sa publication
> doit être faite manuellement si nécessaire.

---

## Maintainer checklist

### Avant toute release

- [ ] Vérifier que les secrets GitHub nécessaires sont valides et non expirés :
      - Core : `ANACONDA_API_TOKEN` (Trusted Publishing PyPI ne nécessite pas de token secret)
      - Plugins : `PYPI_API_TOKEN`, `TEST_PYPI_API_TOKEN`, `ANACONDA_API_TOKEN`, `BOT_TOKEN`
- [ ] Vérifier l'état des services externes (Zenodo, PyPI, Anaconda.org)
- [ ] Lancer les tests CI sur la branche cible
- [ ] Vérifier que le Colab smoke test passe (`install_on_colab.yml`)

### Release du core

- [ ] Vérifier que l'intégration GitHub → Zenodo est active
      (aller sur https://zenodo.org/account/settings/github/ → onglet GitHub →
      `spectrochempy/spectrochempy` doit être **On** ; si besoin,
      toggle Off → On pour forcer la prise en compte)
- [ ] Vérifier que le workflow `build_package.yml` est configuré comme
      Trusted Publisher sur PyPI et TestPyPI (paramètres du projet
      `spectrochempy` sur PyPI → Trusted Publishers → GitHub repository
      `spectrochempy/spectrochempy`, workflow `build_package.yml`,
      environment `pypi`)
- [ ] Lancer **Prepare a new release** avec la version X.Y.Z
- [ ] Vérifier la PR de release (CITATION.cff, zenodo.json, whatsnew)
- [ ] Merger la PR → attendre la Draft Release
- [ ] Vérifier la Draft Release, puis publier
- [ ] Vérifier PyPI : `pip install spectrochempy==X.Y.Z`
- [ ] Vérifier Anaconda : `anaconda show spectrocat/spectrochempy`
- [ ] Vérifier Zenodo : le DOI pointe vers la nouvelle version
- [ ] Vérifier que les docs sont déployées sur `gh-pages`

### Release des plugins

- [ ] Vérifier que `PYPI_API_TOKEN` et `TEST_PYPI_API_TOKEN` sont valides
- [ ] Vérifier que `BOT_TOKEN` est valide (expire tous les 3 mois)
- [ ] Désactiver l'intégration GitHub → Zenodo
- [ ] Lancer **Release an official plugin** avec `confirm_zenodo_disabled=true`
- [ ] Vérifier que le workflow termine sans erreur (commit sauté si version déjà à jour)
- [ ] Vérifier que le flag "Latest" a bien été désactivé sur la Release GitHub du plugin
      (aller sur https://github.com/spectrochempy/spectrochempy/releases →
      vérifier que la release plugin n'affiche pas le badge "Latest")
- [ ] Vérifier PyPI : `pip install spectrochempy-XXX==X.Y.Z`
- [ ] Vérifier Anaconda : `anaconda show spectrocat/spectrochempy-XXX`
- [ ] Répéter pour chaque plugin (nmr → iris → tensor → hypercomplex → carroucell)
- [ ] Réactiver l'intégration GitHub → Zenodo (avant la prochaine release core)

### TestPyPI cleanup

- [ ] Les pushes sur `master` publient automatiquement le core sur TestPyPI
- [ ] Les publications plugins vers TestPyPI ne remplacent pas les versions
      existantes (le workflow utilise `skip-existing: true`)
- [ ] Si une version a été publiée sur TestPyPI puis modifiée, supprimer
      manuellement l'ancienne version sur
      [TestPyPI](https://test.pypi.org/manage/projects/)
- [ ] Ne pas confondre TestPyPI et PyPI lors des vérifications

### Colab verification

- [ ] Le workflow `install_on_colab.yml` ne s'exécute plus automatiquement
      sur les branches (`push` supprimé) — uniquement sur les PR marquées
      `needs-colab` ou en `workflow_dispatch` manuel
- [ ] Avant une release, lancer le `workflow_dispatch` manuel depuis
      **`master`** (ou la branche courante de release) pour valider la
      compatibilité Colab
- [ ] Les deux modes (`core-only` et `with-plugins`) doivent passer
- [ ] En cas d'échec, vérifier les dépendances Colab (numpy, matplotlib, etc.)
      et les contraintes réseau

### Zenodo / plugins

- [ ] Ne jamais laisser Zenodo actif pendant une release plugin
- [ ] Vérifier qu'aucune entrée Zenodo parasite n'a été créée après une
      release plugin
- [ ] Si des entrées plugins existent dans Zenodo, les supprimer
      (voir `emergency-recovery.md`)

---

## TODO — Documentation modulaire (chantier futur)

- Séparer plus clairement les docs `latest`, les docs stables et les docs
  plugins
- Éviter de reconstruire inutilement des versions inchangées (build complet
  même quand seuls quelques fichiers RST ont changé)
- Rendre le version selector moins dépendant des détails de tagging
  (actuellement lié aux répertoires `X.Y.Z` dans le HTML)
