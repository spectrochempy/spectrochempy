document.addEventListener("DOMContentLoaded", function () {
    const versionsDropdown = document.getElementById("versions-dropdown");

    // Fetch the available versions from a JSON file
    fetch("/_static/versions.json")
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
