document.addEventListener("DOMContentLoaded", function () {
    const versionsDropdown = document.getElementById("versions-dropdown");
    
    // Get current path components excluding empty strings
    const pathParts = window.location.pathname.split('/').filter(part => part.length > 0);
    
    // Find current version - first path component if it matches version pattern
    // otherwise it's the root/latest version
    const versionPattern = /^\d+\.\d+\.\d+$/;
    const currentVersion = pathParts.length > 0 && versionPattern.test(pathParts[0]) 
        ? pathParts[0] 
        : 'latest';

    // Get versions from data attribute, which is now statically set for each version
    const versions = (document.documentElement.dataset.versions || '').split(',');

    // Remove empty strings and sort versions in descending order
    const sortedVersions = versions.filter(v => v).sort((a, b) => {
        const partsA = a.split('.').map(Number);
        const partsB = b.split('.').map(Number);
        for (let i = 0; i < 3; i++) {
            if (partsA[i] !== partsB[i]) {
                return partsB[i] - partsA[i];
            }
        }
        return 0;
    });

    // Always add latest version first (root)
    const latestOption = document.createElement("option");
    latestOption.value = window.location.origin + '/';
    latestOption.textContent = "latest";
    latestOption.selected = currentVersion === 'latest';
    versionsDropdown.appendChild(latestOption);
    
    // Add older versions
    sortedVersions.forEach(version => {
        const option = document.createElement("option");
        option.value = `${window.location.origin}/${version}/`;
        option.textContent = version;
        option.selected = currentVersion === version;
        versionsDropdown.appendChild(option);
    });

    // Handle version selection
    versionsDropdown.addEventListener("change", function() {
        const selectedVersion = this.value;
        if (selectedVersion) {
            window.location.href = selectedVersion;
        }
    });
});
