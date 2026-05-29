# Procédure de publication

## Prérequis

### Secrets GitHub nécessaires

Les secrets suivants doivent être définis dans
[Settings → Secrets and variables → Actions](https://github.com/spectrochempy/spectrochempy/settings/secrets/actions)
du dépôt `spectrochempy/spectrochempy` :

| Secret | Usage |
|--------|-------|
| `PYPI_API_TOKEN` | Publication sur PyPI (via `pypa/gh-action-pypi-publish`) |
| `TEST_PYPI_API_TOKEN` | Publication sur Test PyPI (poussées vers master) |
| `ANACONDA_API_TOKEN` | Publication sur Anaconda.org (compte `spectrocat`) |
| `BOT_TOKEN` | PAT personnel utilisé pour contourner la protection de branche lors des releases de plugins (expire tous les 3 mois — penser à le renouveler et mettre à jour le secret) |

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

- Le dépôt `spectrochempy/spectrochempy` est bien activé dans
  [Zenodo GitHub settings](https://zenodo.org/account/settings/github/)
- L'intégration GitHub est active (pas de croix rouge)
- L'onglet **Errors** de la page Zenodo ne contient pas d'erreur active
- `CITATION.cff` et `zenodo.json` sont valides (vérifier les versions)

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

### Déroulement

1. Le workflow `release_plugin.yml` :
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
   - Aller dans
     [Zenodo GitHub settings](https://zenodo.org/account/settings/github/)
   - Décocher / désactiver le dépôt `spectrochempy/spectrochempy`
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
   - Aller dans
     [Zenodo GitHub settings](https://zenodo.org/account/settings/github/)
   - Réactiver le dépôt `spectrochempy/spectrochempy`
   - Vérifier que l'intégration est active (pas de croix rouge)

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

- [ ] Vérifier que les secrets GitHub (`PYPI_API_TOKEN`, `ANACONDA_API_TOKEN`,
      `BOT_TOKEN`) sont valides et non expirés
- [ ] Vérifier l'état des services externes (Zenodo, PyPI, Anaconda.org)
- [ ] Lancer les tests CI sur la branche cible
- [ ] Vérifier que le Colab smoke test passe (`install_on_colab.yml`)

### Release du core

- [ ] Vérifier que l'intégration GitHub → Zenodo est active
- [ ] Lancer **Prepare a new release** avec la version X.Y.Z
- [ ] Vérifier la PR de release (CITATION.cff, zenodo.json, whatsnew)
- [ ] Merger la PR → attendre la Draft Release
- [ ] Vérifier la Draft Release, puis publier
- [ ] Vérifier PyPI : `pip install spectrochempy==X.Y.Z`
- [ ] Vérifier Anaconda : `anaconda show spectrocat/spectrochempy`
- [ ] Vérifier Zenodo : le DOI pointe vers la nouvelle version
- [ ] Vérifier que les docs sont déployées sur `gh-pages`

### Release des plugins

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
