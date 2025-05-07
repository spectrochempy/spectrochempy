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

        const versionPattern = /^\d+\.\d+\.\d+$/;
        return pathParts.length > 0 && versionPattern.test(pathParts[0])
            ? pathParts[0]
            : 'latest';
    }

    const currentVersion = getCurrentVersion();
    const versions = (document.documentElement.dataset.versions || '').split(',');

    // Remove empty strings and sort versions in descending order
    const sortedVersions = versions
        .filter(v => v)
        .sort((a, b) => {
            const partsA = a.split('.').map(Number);
            const partsB = b.split('.').map(Number);
            for (let i = 0; i < 3; i++) {
                if (partsA[i] !== partsB[i]) {
                    return partsB[i] - partsA[i];
                }
            }
            return 0;
        });

    // Add latest version
    const latestOption = document.createElement("option");
    latestOption.value = window.location.origin + basePath;
    latestOption.textContent = "latest";
    latestOption.selected = currentVersion === 'latest';
    versionsDropdown.appendChild(latestOption);

    // Add older versions
    sortedVersions.forEach(version => {
        const option = document.createElement("option");
        option.value = `${window.location.origin}${basePath}${version}/`;
        option.textContent = version;
        option.selected = currentVersion === version;
        versionsDropdown.appendChild(option);
    });

    // Handle version selection
    versionsDropdown.addEventListener("change", function () {
        const selectedVersion = this.value;
        if (selectedVersion) {
            window.location.href = selectedVersion;
        }
    });
});
