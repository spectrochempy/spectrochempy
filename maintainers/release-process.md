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
4. Vérifier que le bouton est sur **Enabled** (vert) — pas grisé (Disabled)
5. Si le dépôt est grisé, cliquer sur le bouton pour le réactiver
6. Si le dépôt est déjà Enabled mais que l'intégration semble ne pas
   fonctionner (par exemple après une phase de releases plugins), on peut
   **toggle** (Disabled → Enabled) pour forcer Zenodo à reconnaître le dépôt
7. Vérifier l'onglet **Errors** (deuxième onglet) : aucune erreur active
   (pas de croix rouge)
8. Vérifier que `CITATION.cff` et `zenodo.json` sont valides (les versions
   sont correctes)

**Avant une release de plugins :**

- Vérifier que l'intégration GitHub est **désactivée** dans Zenodo
  (voir [Zenodo and plugin releases](#zenodo-and-plugin-releases))

### Anaconda.org

- L'organisation `spectrocat` contient le package attendu :
  ```bash
  anaconda show spectrocat/spectrochempy
  ```
  ```bash
  anaconda show spectrocat/spectrochempy-nmr
  anaconda show spectrocat/spectrochempy-iris
  anaconda show spectrocat/spectrochempy-hypercomplex
  anaconda show spectrocat/spectrochempy-carroucell
  ```

- **Première release d'un plugin** : si le package n'existe pas encore sur
  Anaconda, la commande `anaconda show` échouera — c'est normal. Le
  workflow `build_package.yml` utilise une commande `anaconda show` en
  diagnostic avant l'upload. Si le package n'existe pas encore, cette
  commande peut échouer et bloquer le script à cause de `set -e`.

  → Solution : soit supprimer la ligne `anaconda show` du workflow pour
  les plugins, soit créer le package vide manuellement avant la première
  release (`anaconda upload --skip-existing -l main <fichier>.conda`).

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
git checkout upstream/master
git status
```

La branche doit être propre et à jour. Les checks CI doivent tous passer.

### 2. Lancer le workflow

Depuis l'interface GitHub :

1. Aller dans **Actions** → **Prepare a new release**
2. Cliquer **Run workflow**
3. Renseigner le paramètre :

```
versionString = X.Y.Z
```

(par exemple `0.9.0` ou `0.10.0`)

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
- Titre : `SpectroChemPy v.X.Y.Z`

Aller sur la
[page des releases](https://github.com/spectrochempy/spectrochempy/releases)
pour vérifier la Draft.

### 7. Publier la Release GitHub

- Éditer la Draft Release si nécessaire (ajouter des notes)
- Cliquer **Publish release**
- La publication déclenche automatiquement le workflow
  **Build and publish packages** qui publie sur :

  - **PyPI** (label stable, sans `--force`)
  - **Anaconda.org** (label `main`, sans `--force`)
  - **Zenodo** (via l'intégration GitHub)

- La publication déclenche également le workflow **Docs** (`build_docs.yml`)
  via l'événement `release: [published]` :

  > **Note sur la construction de la documentation** : ce build est volontaire.
  > Il vérifie que la documentation de release peut être construite avec le
  > tag publié (`spectrochempy-vX.Y.Z`). Il alimente aussi la documentation
  > versionnée (accessible sous `/<version>/`) et le dropdown des versions.
  > **Ne pas supprimer ce job** dans le workflow `build_docs.yml`.

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

---

## Décider si un plugin nécessite une release

Avant de publier un plugin, comparer les changements depuis son dernier
tag publié.

### Trouver le dernier tag

```bash
git tag --list 'spectrochempy-nmr-v*' --sort=-v:refname
git log --oneline spectrochempy-nmr-v0.1.1..HEAD -- plugins/spectrochempy-nmr
```

### Vérifier la dernière version publiée

```bash
pip index versions spectrochempy-nmr
anaconda show spectrocat/spectrochempy-nmr
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

Depuis **Actions** → **Release an official plugin**, exécuter le workflow avec les
paramètres :

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
   - Vérifie que le plugin est dans la liste officielle
   - Bump la version dans `pyproject.toml` et `recipe.yaml`
   - Pousse le commit sur `master` (via `BOT_TOKEN`)
   - Crée le tag `spectrochempy-XXX-vX.Y.Z`
   - Crée une Release GitHub
2. La Release GitHub déclenche automatiquement :
   - `publish_plugins.yml` → publication **PyPI**
   - `build_package.yml` → publication **Anaconda.org** (label `main`)

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
   - Basculer le bouton sur **Disabled** (le dépôt passe en grisé)
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
   - Basculer le bouton sur **Enabled**
   - Vérifier que l'intégration est active (pas de croix rouge)

> **Rappel** : l'état Zenodo doit toujours être **Enabled** pendant une
> release du core et **Disabled** pendant une release de plugins.
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
      `spectrochempy/spectrochempy` doit être **Enabled** ; si besoin,
      toggle Disabled → Enabled pour forcer la prise en compte)
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
- [ ] Vérifier PyPI : `pip install spectrochempy-XXX==X.Y.Z`
- [ ] Vérifier Anaconda : `anaconda show spectrocat/spectrochempy-XXX`
- [ ] Répéter pour chaque plugin (nmr → iris → hypercomplex → carroucell)
- [ ] Réactiver l'intégration GitHub → Zenodo (avant la prochaine release core)

### TestPyPI cleanup

- [ ] Les pushes sur `master` publient automatiquement sur TestPyPI
- [ ] Les releases plugins sur TestPyPI ne remplacent pas les versions
      existantes (le workflow utilise `skip-existing: true`)
- [ ] Si une version a été publiée sur TestPyPI puis modifiée, supprimer
      manuellement l'ancienne version sur
      [TestPyPI](https://test.pypi.org/manage/projects/)
- [ ] Ne pas confondre TestPyPI et PyPI lors des vérifications

### Colab verification

- [ ] Le workflow `install_on_colab.yml` s'exécute automatiquement sur les PR
      marquées `needs-colab`
- [ ] Avant une release, vérifier que le test Colab passe en
      `workflow_dispatch` manuel
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
  (actuellement lié aux répertoires `X.Y.Z` dans le HTML et aux alias de
  tags locaux)
