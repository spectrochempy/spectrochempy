(() => {
  let enabled = false;

  document.addEventListener("readthedocs-addons-data-ready", (event) => {
    enabled = event.detail.data()?.addons?.search?.enabled === true;
  });

  function openSearch(event) {
    if (!enabled || !document.querySelector("readthedocs-search")) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    document.dispatchEvent(new CustomEvent("readthedocs-search-show"));
  }

  document.addEventListener("click", (event) => {
    if (event.target.closest(".search-button__button")) openSearch(event);
  }, true);

  document.addEventListener("focusin", (event) => {
    if (event.target.matches(".bd-search input")) openSearch(event);
  }, true);

  window.addEventListener("keydown", (event) => {
    if (!event.shiftKey && !event.altKey &&
        (event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
      openSearch(event);
    }
  }, true);
})();
