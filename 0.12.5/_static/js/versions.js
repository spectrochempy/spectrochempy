document.addEventListener("DOMContentLoaded", function () {
    const versionsDropdown = document.getElementById("versions-dropdown");

    // Get the base URL path by analyzing the current location
    function getBasePath() {
        const path = window.location.pathname;

        // Check if we're on the main site (spectrochempy.fr)
        if (window.location.hostname === 'www.spectrochempy.fr') {
            return '/';
        }

        // For GitHub Pages or other hosts, extract the base path
        // Example: for fernandezc.github.io/spectrochempy/1.2.3/
        // we want /spectrochempy/
        const parts = path.split('/');
        if (parts.length >= 2) {
            // Look for the first part that could be a version number
            for (let i = 1; i < parts.length; i++) {
                if (parts[i] === 'spectrochempy') {
                    return '/' + parts[i] + '/';
                }
            }
        }

        // Default to root if no specific path is found
        return '/';
    }

    const basePath = getBasePath();

    // Get current version from URL
    function getCurrentVersion() {
        const pathParts = window.location.pathname
            .replace(basePath, '/')
            .split('/')
            .filter(part => part.length > 0);

        const versionPattern = /^\d+\.\d+\.\d+(?:rc\d+)?$/;
        return pathParts.length > 0 && versionPattern.test(pathParts[0])
            ? pathParts[0]
            : 'latest';
    }

    function parseVersion(v) {
        const match = v.match(/^(\d+)\.(\d+)\.(\d+)(?:rc(\d+))?$/);
        if (!match) {
            return [0, 0, 0, Infinity];
        }
        const rc = match[4] === undefined ? Infinity : Number(match[4]);
        return [Number(match[1]), Number(match[2]), Number(match[3]), rc];
    }

    function sortVersions(versions) {
        return versions
            .filter(v => typeof v === 'string' && v)
            .sort((a, b) => {
                const partsA = parseVersion(a);
                const partsB = parseVersion(b);
                for (let i = 0; i < partsA.length; i++) {
                    if (partsA[i] !== partsB[i]) {
                        return partsB[i] - partsA[i];
                    }
                }
                // A final release sorts above its own release candidate
                if (a.includes('rc') !== b.includes('rc') && partsA[0] === partsB[0]
                    && partsA[1] === partsB[1] && partsA[2] === partsB[2]) {
                    return a.includes('rc') ? 1 : -1;
                }
                return 0;
            });
    }

    async function loadVersions() {
        const fallback = (document.documentElement.dataset.versions || '').split(',');
        try {
            const response = await fetch(`${window.location.origin}${basePath}_static/versions.json`, {
                cache: 'no-cache',
            });
            if (!response.ok) {
                return fallback;
            }
            const manifest = await response.json();
            if (Array.isArray(manifest.versions)) {
                return manifest.versions;
            }
            if (Array.isArray(manifest)) {
                return manifest
                    .map(item => typeof item === 'string' ? item : item?.name)
                    .filter(name => /^\d+\.\d+\.\d+(?:rc\d+)?$/.test(name));
            }
            return fallback;
        } catch {
            return fallback;
        }
    }

    function populateDropdown(versions) {
        const currentVersion = getCurrentVersion();
        const sortedVersions = sortVersions(versions);

        const latestOption = document.createElement("option");
        latestOption.value = window.location.origin + basePath;
        latestOption.textContent = "latest";
        latestOption.selected = currentVersion === 'latest';
        versionsDropdown.appendChild(latestOption);

        sortedVersions.forEach(version => {
            const option = document.createElement("option");
            option.value = `${window.location.origin}${basePath}${version}/`;
            option.textContent = version;
            option.selected = currentVersion === version;
            versionsDropdown.appendChild(option);
        });
    }

    loadVersions().then(populateDropdown);

    versionsDropdown.addEventListener("change", function () {
        const selectedVersion = this.value;
        if (selectedVersion) {
            window.location.href = selectedVersion;
        }
    });
});
