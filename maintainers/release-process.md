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

### Comptes externes

- **Zenodo** : l'intégration GitHub → Zenodo doit être activée sur le dépôt
  ([instructions Zenodo](https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content)).
  Une Release GitHub publiée déclenche automatiquement l'archivage DOI.

---

## Release du core

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

### Workflow

Depuis **Actions** → **Release an official plugin**, exécuter le workflow avec les
paramètres :

```
plugin_name: spectrochempy-XXX
version: X.Y.Z
```

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

## Ordre recommandé

1. **Release du core** → attendre la fin des builds CI
2. **Vérifier PyPI** : `pip install spectrochempy==X.Y.Z`
3. **Vérifier Anaconda** : `anaconda show spectrocat/spectrochempy`
4. **Vérifier Zenodo** : le DOI doit pointer vers la nouvelle version
5. **Release des plugins** (dans cet ordre) :
   - `spectrochempy-nmr`
   - `spectrochempy-iris`
   - `spectrochempy-hypercomplex`
   - `spectrochempy-carroucell`

> **Note** : `spectrochempy-cantera` est actuellement **expérimental**
> et n'est pas publié automatiquement par les workflows CI. Sa publication
> doit être faite manuellement si nécessaire.
