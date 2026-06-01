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

## Le push vers master échoue dans le workflow **Release an official plugin** (`release_plugin.yml`)

### Symptôme

```
remote: error: GH006: Protected branch update failed for refs/heads/master.
remote: - Changes must be made through a pull request.
```

### Cause

Le `GITHUB_TOKEN` automatique ne peut pas contourner les règles de
protection de la branche `master`. Le workflow **Release an official plugin** (`release_plugin.yml`)
   utilise `secrets.BOT_TOKEN` (un PAT personnel) pour le checkout, ce qui
permet de pusher.

Si ce secret est expiré ou a été révoqué, le push est rejeté.

### Résolution

1. Créer un nouveau **classic PAT** sur
   [github.com/settings/tokens](https://github.com/settings/tokens) avec le
   scope `repo`
2. Mettre à jour le secret `BOT_TOKEN` dans
   [Settings → Secrets → Actions](https://github.com/spectrochempy/spectrochempy/settings/secrets/actions)
3. Relancer le workflow

> **Note** : Les PAT arrivent à expiration après 3 mois. Ajouter un rappel
> calendaire pour le renouvellement.

---

## Erreur setuptools-scm / version incompatible

### Symptôme

```text
Could not find a version that satisfies the requirement spectrochempy>=0.9
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

## Publication PyPI échouée — core (`build_package.yml`)

### Symptôme

Le job `build-and-publish_pypi` échoue dans le workflow
`build_package.yml` (package **core**).

### Vérifications

1. **Trusted Publishing / OIDC** : le package core utilise
   [Trusted Publishing](https://docs.pypi.org/trusted-publishers/).
   Il n'utilise **pas** `PYPI_API_TOKEN` ou `TEST_PYPI_API_TOKEN`.
   Vérifier la configuration sur PyPI/TestPyPI :
   - Aller sur les paramètres du projet `spectrochempy` sur PyPI
   - Vérifier que le workflow `build_package.yml` du dépôt
     `spectrochempy/spectrochempy` est bien listé dans Trusted Publishers
   - L'environment `pypi` doit correspondre à celui du workflow
2. **Permissions GitHub** : le workflow nécessite `contents: read`,
   `packages: write` et `id-token: write` dans les permissions du
   workflow
3. **Version déjà publiée** : le workflow publie la release sur PyPI sans
   `--skip-existing`. Si une version identique existe déjà (ex. retry),
   le build échoue. Dans ce cas, supprimer la version sur PyPI ou
   incrémenter le numéro de version

### Re-running a stable PyPI release

Stable PyPI releases cannot overwrite an already published version. If a release
workflow is re-run after the package version has already been uploaded to PyPI,
the PyPI upload step is expected to fail.

Do not treat this as a packaging regression. Either:

- create a new release/tag with a new version, or
- skip the PyPI upload if the existing artifact is already correct.

This differs from some TestPyPI/dev workflows where ``skip-existing`` may be used
to avoid hard failures during repeated test uploads.

### Résolution manuelle

```bash
# Vérifier si la version existe déjà sur PyPI
pip index versions spectrochempy

# Si nécessaire, forcer la publication depuis une machine locale (token API nécessaire)
python -m build
python -m twine upload dist/*
```

---

## Publication PyPI échouée — plugins (`publish_plugins.yml`)

### Symptôme

Le job `build-and-publish_plugins` échoue dans le workflow
`publish_plugins.yml`.

### Vérifications

1. **Token PyPI / TestPyPI** : les plugins utilisent les **API tokens** :
   - `PYPI_API_TOKEN` pour PyPI
   - `TEST_PYPI_API_TOKEN` pour TestPyPI
   Vérifier qu'ils sont définis dans les
   [secrets du dépôt](https://github.com/spectrochempy/spectrochempy/settings/secrets/actions)
   et qu'ils n'ont pas expiré ou été révoqués
2. **Version déjà publiée** : le workflow publie les plugins avec
   `skip-existing: true` sur TestPyPI. Sur PyPI stable, une version
   déjà publiée provoque un échec (pas de `--skip-existing`).

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
   - Sur release : upload vers le label `main` (avec `--force`, pour déplacer
     si nécessaire une build déjà publiée sur `dev` vers le label stable)
   - Sur push : upload vers le label `dev` (avec `--force`)
4. **Version déjà publiée** : si Anaconda refuse malgré tout l'upload,
   supprimer la version sur Anaconda si nécessaire :
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

## Build conda des plugins échoué

### Symptôme

```
Error: × Test failed: failed to setup test environment:
  │ └─ spectrochempy-iris ==0.1.1 cannot be installed because
  │    └─ spectrochempy >=0.9,<0.10, which cannot be installed
```

Le plugin est publié sur PyPI mais pas sur Anaconda.

### Causes possibles

**1. Version du core mal détectée pendant une release plugin**

Lors d'une release plugin (tag `spectrochempy-nmr-v0.1.1`),
`setuptools_scm` peut détecter la version `0.1.1` (celle du tag plugin)
au lieu de la version `0.9.0` du core. Le package core est alors
buildé avec une version incorrecte.

*Résolution* : le workflow `build_package.yml` inclut maintenant une
étape "Determine core version for conda build" qui extrait la version
depuis le dernier tag `spectrochempy-v*`.

**2. Strict channel priority exclut spectrocat**

Le solveur conda utilise `conda-forge` en priorité. Comme
`spectrochempy` existe sur `conda-forge` (à une version antérieure),
le solveur exclut les versions plus récentes de `spectrocat` à cause
de la priorité stricte.

*Résolution* : l'ordre des canaux dans le build des plugins a été
inversé : `spectrocat/label/dev` (pour le core dev construit sur push) et
`spectrocat` passent avant `conda-forge`. Les plugins eux-mêmes ne sont
uploadés sur Anaconda.org que pendant une release stable de plugin.

### Résolution pour une release déjà publiée

Si le plugin est déjà publié sur PyPI mais pas sur Anaconda :

1. Supprimer la Release GitHub et le tag du plugin :
   ```bash
   gh release delete spectrochempy-XXX-vX.Y.Z --repo spectrochempy/spectrochempy --yes
   git push upstream --delete refs/tags/spectrochempy-XXX-vX.Y.Z
   ```
2. Relancer le workflow **Release an official plugin** depuis
   [Actions](https://github.com/spectrochempy/spectrochempy/actions/workflows/release_plugin.yml)

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

---

## Zenodo affiche "Failed" après une release

### Symptôme

Une release apparaît dans Zenodo avec le statut :

```text
Failed
```

et le message :

```text
Bad credentials
```

Les releases suivantes peuvent être synchronisées avec succès après
reconnexion, mais la release en échec reste bloquée.

### Cause

Les identifiants GitHub stockés par Zenodo sont devenus invalides
(expiration, révocation, changement de mot de passe…). Zenodo n'arrive
plus à cloner le dépôt ni à importer la release.

### Résolution

1. Aller dans **Zenodo → Account → GitHub** :
   https://zenodo.org/account/settings/github/
2. **Déconnecter** GitHub (Disconnect).
3. **Reconnecter** GitHub (Connect) — autoriser l'accès.
4. **Réactiver** le dépôt `spectrochempy/spectrochempy` si nécessaire.
5. Vérifier que les nouvelles releases sont correctement synchronisées.
6. Si la release en échec n'est pas retraitée automatiquement :

   * recréer la GitHub Release concernée (en conservant le tag existant) ;
   * la republier.

Zenodo importe alors la release et génère le DOI normalement.

### Notes

- Une synchronisation réussie des releases suivantes **ne garantit pas**
  que la release en échec sera retraitée automatiquement.
- La recréation de la GitHub Release peut être nécessaire pour forcer
  la réimportation.

---

## Zenodo a archivé des releases plugins par erreur

### Symptômes

- Des enregistrements Zenodo apparaissent avec des versions de plugins
  (ex. `spectrochempy-carroucell-v0.1.1`) dans la liste des enregistrements
  de la communauté SpectroChemPy
- Le titre de l'enregistrement Zenodo affiche "SpectroChemPy..." mais la
  version est celle d'un plugin
- Les métadonnées (auteurs, description) sont celles du core, pas du
  plugin
- Des DOI ont été créés pour des releases plugins qui ne devraient pas
  être archivées

### Cause

L'intégration GitHub → Zenodo étant active sur
`spectrochempy/spectrochempy`, Zenodo archive automatiquement **toutes**
les GitHub Releases, y compris celles des plugins (tags
`spectrochempy-XXX-vX.Y.Z`).

### Résolution

1. **Supprimer les enregistrements Zenodo erronés** :
   - Aller sur la
     [page Zenodo de SpectroChemPy](https://zenodo.org/communities/spectrochempy)
   - Repérer les entrées correspondant aux releases plugins
   - Ouvrir chaque enregistrement → cliquer sur **Edit** → **Delete**
     (ou contacter l'admin Zenodo si le bouton de suppression n'est pas
     disponible)
   - Confirmer la suppression

2. **Désactiver l'intégration GitHub dans Zenodo** avant toute future
   release plugin :
   - Aller dans
     [Zenodo GitHub settings](https://zenodo.org/account/settings/github/)
   - Décocher / désactiver le dépôt `spectrochempy/spectrochempy`
   - Ne réactiver que pour les releases du core

3. **Vérifier qu'aucune nouvelle entrée parasite n'est créée** après les
   prochaines releases plugins.

### Prévention

- Suivre la procédure décrite dans
  [release-process.md — Zenodo and plugin releases](../release-process.md#zenodo-and-plugin-releases)
- Ne jamais laisser l'intégration Zenodo active pendant une phase de
  release plugin

### Note stratégique

À long terme, si les plugins doivent avoir leurs propres DOI, deux
options :

1. **Dépôts séparés** : déplacer chaque plugin dans son propre dépôt
   GitHub et connecter Zenodo indépendamment
2. **Procédure manuelle** : créer les entrées Zenodo manuellement pour
   chaque plugin, avec des métadonnées spécifiques

Sinon, les plugins ne doivent pas créer d'entrées Zenodo séparées — seul
le core a un DOI via l'intégration GitHub automatique.
