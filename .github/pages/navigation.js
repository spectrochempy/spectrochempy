document.addEventListener("DOMContentLoaded", async () => {
  const script = [...document.scripts].find(item => item.src.endsWith("/pages-navigation.js"));
  if (!script) return;
  const root = new URL(".", script.src);
  const parts = location.pathname.slice(root.pathname.length).split("/");
  const preview = parts[0] === "previews";
  const current = parts.splice(0, preview ? 2 : 1).join("/");
  const nav = document.createElement("nav");
  nav.setAttribute("aria-label", "Documentation projects and versions");
  nav.style.cssText = "padding:.5rem 1rem;display:flex;gap:1rem;flex-wrap:wrap;align-items:center;border-bottom:1px solid #888";
  const label = document.createElement("label");
  label.textContent = preview ? `PR preview (${current}) ` : "Documentation version ";
  const select = document.createElement("select");
  select.setAttribute("aria-label", "Documentation version");
  select.add(new Option(current, current, true, true));
  label.append(select);
  nav.append(label);
  let stableVersion = null;
  try {
    const response = await fetch(new URL("versions.json", root), {cache: "no-cache"});
    if (!response.ok) throw new Error("Manifest unavailable");
    const data = await response.json();
    stableVersion = data.stable;
    for (const version of data.versions) {
      if (version.path !== current) select.add(new Option(version.name, version.path));
    }
    document.querySelectorAll("[data-docs-target='stable']").forEach(link => {
      link.href = new URL(data.stable ? "stable/" : "latest/", root).href;
    });
    const configResponse = await fetch(new URL("projects.json", root), {cache: "no-cache"});
    if (configResponse.ok) {
      for (const project of await configResponse.json()) {
        const url = new URL(project.url);
        if (url.protocol !== "https:") continue;
        const link = document.createElement("a");
        link.href = url.href;
        link.textContent = project.name;
        nav.append(link);
      }
    }
  } catch (error) {
    console.warn("Documentation navigation unavailable", error);
  }
  select.addEventListener("change", async () => {
    const version = select.value === "stable" && stableVersion ? stableVersion : select.value;
    const fallback = new URL(`${version}/`, root);
    let destination = new URL(parts.join("/"), fallback);
    try {
      if (!(await fetch(destination, {method: "HEAD"})).ok) destination = fallback;
    } catch { destination = fallback; }
    destination.hash = location.hash;
    destination.search = location.search;
    location.assign(destination.href);
  });
  document.body.prepend(nav);
  document.querySelectorAll(".scp-version-switcher, #versions-dropdown").forEach(element => element.hidden = true);
});
