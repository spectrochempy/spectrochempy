document.addEventListener("DOMContentLoaded", function () {
    const versionsDropdown = document.getElementById("versions-dropdown");

    // Construct the correct path to the versions.json file
    const basePath = window.location.pathname.split('/').slice(0, window.location.pathname.split('/').indexOf('spectrochempy') + 1).join('/');
    const versionsPath = `${window.location.origin}${basePath}/_static/versions.json`;

    // Fetch the available versions from the JSON file
    fetch(versionsPath)
        .then(response => response.json())
        .then(data => {
            data.versions.forEach(version => {
                const option = document.createElement("option");
                option.value = version.url;
                option.textContent = version.name;
                versionsDropdown.appendChild(option);
            });
        })
        .catch(error => console.error("Error fetching versions:", error));

    // Add event listener to handle version change
    versionsDropdown.addEventListener("change", function () {
        const selectedVersion = versionsDropdown.value;
        if (selectedVersion) {
            window.location.href = selectedVersion;
        }
    });
});
