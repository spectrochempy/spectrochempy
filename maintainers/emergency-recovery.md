# Incidents et résolutions

Ce document recense les incidents déjà rencontrés lors des processus de
release, avec leur cause et la résolution appliquée.

---

## Les checks ne démarrent pas sur une PR de release

### Symptôme

Sur une Pull Request de release créée par `github-actions[bot]`, les
workflows GitHub ne se lancent pas et affichent :

```
Expected — Waiting for status to be reported
```

### Cause

Les workflows déclenchés par `GITHUB_TOKEN` (jeton d'authentification
automatique) ne déclenchent pas de nouveaux événements GitHub,
conformément à la
[sécurité GitHub](https://docs.github.com/en/actions/security-for-github-actions/security-guides/automatic-token-authentication#using-the-github_token-in-a-workflow).
Les PR créées par `github-actions[bot]` héritent de ce comportement, donc
les workflows de CI (tests, builds) ne démarrent pas.

### Résolution

- Vérifier manuellement les modifications dans la PR
- Si tout est correct, **merge manuellement** la PR (les checks sont requis
  mais peuvent être contournés par un administrateur)
- Une fois mergée, les workflows post-merge fonctionneront normalement
  (push event via `GITHUB_TOKEN`)

### Prévention

- Utiliser un **PAT dédié** (Personal Access Token) pour le checkout dans
  `prepare_new_release.yml`, ce qui déclencherait les événements GitHub
  normalement

---

## Erreur setuptools-scm / version incompatible

### Symptôme

```text
Could not find a version that satisfies the requirement spectrochempy>=0.9.0.dev0
```

### Cause

La version locale de `spectrochempy` calculée par `setuptools_scm` est
trop ancienne. Cela arrive si :

- Le tag le plus récent dans le dépôt local est trop vieux
- Le checkout n'a pas récupéré tous les tags (`fetch-depth: 0` manquant)

### Résolution

```bash
# Vérifier la version calculée localement
python -c "import spectrochempy; print(spectrochempy.__version__)"

# Forcer une version de développement si nécessaire
export SETUPTOOLS_SCM_PRETEND_VERSION=X.Y.Z.dev0

# Vérifier que tous les tags sont récupérés
git fetch --tags upstream
```

Si le problème vient des dépendances des plugins, s'assurer que le
versionnage dans `recipe.yaml` du plugin est compatible avec la version
du core publiée (ex. `spectrochempy >=0.9.0,<0.10`).

---

## Publication PyPI échouée

### Symptôme

Le job `build-and-publish_pypi` échoue dans le workflow
`build_package.yml`.

### Vérifications

1. **Token PyPI** : vérifier que `PYPI_API_TOKEN` est défini dans les
   [secrets du dépôt](https://github.com/spectrochempy/spectrochempy/settings/secrets/actions)
   et qu'il n'a pas expiré ou été révoqué
2. **Permissions GitHub** : le workflow nécessite `contents: read`,
   `packages: write` et `id-token: write` dans les permissions du
   workflow
3. **Version déjà publiée** : le workflow publie la release sur PyPI sans
   `--skip-existing`. Si une version identique existe déjà (ex. retry),
   le build échoue. Dans ce cas, supprimer la version sur PyPI ou
   incrémenter le numéro de version

### Résolution

```bash
# Vérifier si la version existe déjà sur PyPI
pip index versions spectrochempy

# Si nécessaire, forcer la publication depuis une machine locale
python -m build
python -m twine upload dist/*
```

---

## Publication Anaconda échouée

### Symptôme

Le job `build_and_publish_conda_package` échoue.

### Vérifications

1. **Token Anaconda** : vérifier que `ANACONDA_API_TOKEN` est défini dans
   les secrets du dépôt et qu'il a les permissions nécessaires sur le
   compte `spectrocat`
2. **Package cible** : vérifier que le package `spectrochempy` existe bien
   dans l'org `spectrocat` :
   ```bash
   anaconda show spectrocat/spectrochempy
   ```
3. **Labels** :
   - Sur release : upload vers le label `main` (sans `--force`)
   - Sur push : upload vers le label `dev` (avec `--force`)
4. **Version déjà publiée** : si le label `main` contient déjà la même
   version, le build échoue (pas de `--force` sur la release). Supprimer
   la version sur Anaconda si nécessaire :
   ```bash
   anaconda remove spectrocat/spectrochempy/X.Y.Z
   ```

### Problème connu : `SETUPTOOLS_SCM_PRETEND_VERSION` hardcodé

- **Symptôme** : le build conda utilise une version en `dev0` au lieu de
  la version release
- **Cause** : la variable d'environnement était durcie dans le workflow
- **Résolution** : supprimer la variable `SETUPTOOLS_SCM_PRETEND_VERSION`
  du step "Generate conda recipe" pour laisser `setuptools_scm` détecter
  la version depuis le tag git

---

## Zenodo non mis à jour

### Symptôme

Après publication d'une Release GitHub, le DOI Zenodo continue de pointer
vers une version antérieure.

### Vérifications

1. **Intégration GitHub → Zenodo** : vérifier que le dépôt
   `spectrochempy/spectrochempy` est bien connecté à Zenodo dans les
   [paramètres Zenodo](https://zenodo.org/account/settings/github/)
2. **Release GitHub publiée** : Zenodo ne se déclenche que sur les
   Releases **publiées** (pas les Drafts)
3. **Tag de release** : le tag doit suivre le format
   `spectrochempy-vX.Y.Z` pour que Zenodo le reconnaisse

### Résolution

- Aller sur la
  [page Zenodo de SpectroChemPy](https://zenodo.org/communities/spectrochempy)
- Si la release est bien publiée mais non archivée, cliquer sur
  **Reserve DOI** manuellement depuis l'interface Zenodo
- Si l'intégration est cassée, la réactiver depuis
  [Zenodo GitHub settings](https://zenodo.org/account/settings/github/)
  (déconnecter puis reconnecter le dépôt)
