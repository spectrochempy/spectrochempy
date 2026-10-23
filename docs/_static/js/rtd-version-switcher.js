(() => {
  function documentationURL(version) {
    try {
      const url = new URL(version.urls.documentation);
      return ["https:", "http:"].includes(url.protocol) ? url.href : null;
    } catch {
      return null;
    }
  }

  function populate(data, localPreview = false) {
    const versions = data?.versions;
    if (!versions?.current || !Array.isArray(versions.active)) return;
    const available = versions.active.filter(
      (version) => version.built && !version.hidden && documentationURL(version),
    );
    if (!available.length) return;
    let populated = false;
    document.querySelectorAll(".scp-version-switcher").forEach((container) => {
      const select = container.querySelector("select");
      const current = versions.current;
      const label = localPreview
        ? container.dataset.buildLabel
        : current.type === "external"
          ? `PR #${current.slug}`
          : container.dataset.buildLabel;
      select.title = localPreview
        ? "Documentation version — local preview"
        : `Documentation version — ${current.slug}`;
      select.replaceChildren(new Option(label, "", true, true));
      available.forEach((version) => {
        if (!localPreview && version.slug === current.slug) return;
        const name = version.display_name || (
          ["latest", "stable"].includes(version.slug)
            ? version.slug
            : version.verbose_name || version.slug
        );
        const option = new Option(name, documentationURL(version));
        option.title = version.slug;
        select.add(option);
      });
      select.disabled = false;
      select.onchange = () => {
        if (select.value) window.location.assign(select.value);
      };
      populated = true;
    });
    if (populated && !localPreview) {
      document.documentElement.classList.add("scp-version-switcher-ready");
    }
  }

  document.addEventListener("readthedocs-addons-data-ready", (event) => {
    populate(event.detail.data());
  });

  const preview = document.querySelector(".scp-version-switcher[data-preview-json]");
  if (preview) {
    fetch(preview.dataset.previewJson)
      .then((response) => {
        if (!response.ok) throw new Error("Version preview unavailable");
        return response.json();
      })
      .then((data) => populate(data, true))
      .catch(() => {});
  }
})();
