"""
Check local links, fragments, and resources in an existing HTML build.

Run after a Sphinx HTML build: ``python docs/tools/check_html_links.py BUILD/html``.
External URLs and deployment-root URLs require a separate online check.
"""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote
from urllib.parse import urljoin
from urllib.parse import urlsplit


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.anchors = set()
        self.links = []
        self.feed(text)

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if attrs.get("id"):
            self.anchors.add(attrs["id"])
        if tag == "a" and attrs.get("name"):
            self.anchors.add(attrs["name"])
        for attribute in ("href", "src", "poster"):
            if attrs.get(attribute):
                self.links.append(attrs[attribute])
        if tag == "object" and attrs.get("data"):
            self.links.append(attrs["data"])


def check_html_links(root):
    root = Path(root).resolve()
    pages = {
        path.relative_to(root).as_posix(): Page(path.read_text(encoding="utf-8"))
        for path in root.rglob("*.html")
        if "_static" not in path.relative_to(root).parts
    }
    if "index.html" not in pages:
        raise ValueError(f"No index.html found in {root}")
    problems = []
    for source, page in sorted(pages.items()):
        for link in sorted(set(page.links)):
            parsed = urlsplit(link)
            if parsed.scheme or parsed.netloc or link.startswith("/"):
                continue
            target = unquote(
                urlsplit(urljoin("https://local.invalid/" + source, link)).path
            ).lstrip("/")
            if not target or target.endswith("/"):
                target += "index.html"
            path = root / target
            if not path.is_file():
                problems.append((source, link, "missing file"))
                continue
            fragment = unquote(parsed.fragment)
            if fragment and target in pages and fragment not in pages[target].anchors:
                problems.append((source, link, "missing anchor"))
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html_directory", type=Path)
    args = parser.parse_args()
    problems = check_html_links(args.html_directory)
    for source, link, reason in problems:
        print(f"{source}: {reason}: {link}")  # noqa: T201
    print(f"{len(problems)} broken local links or resources")  # noqa: T201
    return bool(problems)


if __name__ == "__main__":
    raise SystemExit(main())
