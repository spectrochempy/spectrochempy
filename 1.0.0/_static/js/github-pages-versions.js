document.addEventListener("DOMContentLoaded", () => {
    const selector = document.getElementById("versions-dropdown");
    if (!selector) {
        return;
    }

    const versionPattern = /^\d+\.\d+\.\d+(?:rc\d+)?$/;

    function getBasePath() {
        const marker = "/spectrochempy/";
        const path = window.location.pathname;
        if (window.location.hostname.endsWith("github.io") && path.includes(marker)) {
            return path.slice(0, path.indexOf(marker) + marker.length);
        }
        return "/";
    }

    const basePath = getBasePath();
    const previewPath = (selector.dataset.previewName || "").replaceAll("/", "-");

    function relativePathFromBase() {
        const pathname = window.location.pathname;
        const relative = pathname.startsWith(basePath)
            ? pathname.slice(basePath.length)
            : pathname.replace(/^\/+/, "");
        const parts = relative.split("/").filter(Boolean);
        const first = parts[0] || "";
        if (first === "latest" || first === "dev" || versionPattern.test(first)
            || (previewPath && first === previewPath)) {
            parts.shift();
        }
        if (parts.length === 0) {
            return "index.html";
        }
        return parts.join("/") + (pathname.endsWith("/") ? "/" : "");
    }

    function currentLocation(manifest) {
        const relative = window.location.pathname.startsWith(basePath)
            ? window.location.pathname.slice(basePath.length)
            : window.location.pathname.replace(/^\/+/, "");
        const first = relative.split("/").filter(Boolean)[0] || "";
        if (previewPath && first === previewPath) {
            return {kind: "preview", version: previewPath};
        }
        if (first === "latest" || first === "dev") {
            return {kind: "development", version: manifest.development};
        }
        if (versionPattern.test(first)) {
            return {
                kind: first === manifest.stable ? "stable" : "archived",
                version: first,
            };
        }
        if (selector.dataset.docsContext === "preview") {
            return {kind: "preview", version: previewPath};
        }
        return {kind: "stable", version: manifest.stable};
    }

    function parseVersion(value) {
        const match = value.match(/^(\d+)\.(\d+)\.(\d+)(?:rc(\d+))?$/);
        if (!match) {
            return [0, 0, 0, -1];
        }
        const rc = match[4] === undefined ? Number.MAX_SAFE_INTEGER : Number(match[4]);
        return [Number(match[1]), Number(match[2]), Number(match[3]), rc];
    }

    function sortVersions(versions) {
        return [...new Set(versions)]
            .filter(version => typeof version === "string" && versionPattern.test(version))
            .sort((left, right) => {
                const leftParts = parseVersion(left);
                const rightParts = parseVersion(right);
                for (let index = 0; index < leftParts.length; index += 1) {
                    if (leftParts[index] !== rightParts[index]) {
                        return rightParts[index] - leftParts[index];
                    }
                }
                return 0;
            });
    }

    function fallbackManifest() {
        const versions = (selector.dataset.versions || "").split(",").filter(Boolean);
        const stable = selector.dataset.stableVersion || sortVersions(versions)[0] || "";
        return {development: "latest", stable, versions};
    }

    async function loadManifest() {
        const fallback = fallbackManifest();
        try {
            const manifestUrl = new URL(`${basePath}_static/versions.json`, window.location.origin);
            const response = await fetch(manifestUrl, {cache: "no-cache"});
            if (!response.ok) {
                return fallback;
            }
            const payload = await response.json();
            if (Array.isArray(payload)) {
                const versions = payload
                    .map(item => typeof item === "string" ? item : item?.name)
                    .filter(Boolean);
                return {...fallback, versions};
            }
            return {
                development: payload.development || payload.latest || "latest",
                stable: payload.stable || fallback.stable,
                versions: Array.isArray(payload.versions) ? payload.versions : fallback.versions,
            };
        } catch {
            return fallback;
        }
    }

    function targetRoot(segment) {
        return `${basePath}${segment ? `${segment}/` : ""}`;
    }

    function addOption(parent, label, root, selected) {
        const option = document.createElement("option");
        option.value = root;
        option.textContent = label;
        option.selected = selected;
        parent.appendChild(option);
        return option;
    }

    function populateSelector(manifest) {
        selector.replaceChildren();
        const current = currentLocation(manifest);
        const sorted = sortVersions(manifest.versions);

        if (current.kind === "preview") {
            addOption(
                selector,
                `Preview — ${selector.dataset.previewName || current.version || "pull request"}`,
                targetRoot(previewPath),
                true,
            );
        }

        if (manifest.stable) {
            addOption(
                selector,
                `Stable — ${manifest.stable}`,
                targetRoot(manifest.stable),
                current.kind === "stable",
            );
        }
        addOption(
            selector,
            "Development — unreleased",
            targetRoot(manifest.development || "latest"),
            current.kind === "development",
        );

        const archived = sorted.filter(version => version !== manifest.stable);
        if (current.kind === "archived" && !archived.includes(current.version)) {
            archived.unshift(current.version);
        }
        if (archived.length > 0) {
            const group = document.createElement("optgroup");
            group.label = "Previous versions";
            archived.forEach(version => {
                addOption(
                    group,
                    version,
                    targetRoot(version),
                    current.kind === "archived" && current.version === version,
                );
            });
            selector.appendChild(group);
        }
    }

    async function pageExists(url) {
        try {
            let response = await fetch(url, {method: "HEAD", redirect: "follow"});
            if (response.status === 405) {
                response = await fetch(url, {method: "GET", redirect: "follow"});
            }
            return response.ok;
        } catch {
            return false;
        }
    }

    async function navigateToVersion(root) {
        const relative = relativePathFromBase();
        const candidate = new URL(relative, new URL(root, window.location.origin));
        const destination = await pageExists(candidate)
            ? candidate
            : new URL("index.html", new URL(root, window.location.origin));
        destination.search = window.location.search;
        destination.hash = window.location.hash;
        window.location.assign(destination.href);
    }

    selector.addEventListener("change", () => {
        if (selector.value) {
            navigateToVersion(selector.value);
        }
    });

    document.querySelectorAll("[data-docs-target='stable']").forEach(link => {
        link.addEventListener("click", event => {
            const stable = selector.dataset.stableVersion;
            if (stable) {
                event.preventDefault();
                navigateToVersion(targetRoot(stable));
            }
        });
    });

    loadManifest().then(populateSelector);
});
