"""Download public RBI PDF links and maintain a corpus manifest.

This is a best-effort scraper. RBI pages can change their HTML structure;
if fewer than 60 PDFs are collected, manually download PDFs and add rows
to data/manifest.csv rather than spending too long debugging the scraper.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "manifest.csv"
SOURCES = [
    "https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx",
    "https://www.rbi.org.in/Scripts/NotificationUser.aspx",
]
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
FIELDS = ["file", "title", "date", "source_url", "pages", "parser"]


def pdf_links(page_url: str, html: str) -> list[tuple[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    found: dict[str, str] = {}
    for anchor in soup.find_all("a", href=True):
        href = urljoin(page_url, anchor["href"].strip())
        if ".pdf" in urlparse(href).path.lower():
            title = " ".join(anchor.get_text(" ", strip=True).split())
            found[href] = title or Path(urlparse(href).path).name
    return list(found.items())


def read_existing_urls() -> set[str]:
    if not MANIFEST.exists():
        return set()
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        return {row.get("source_url", "") for row in csv.DictReader(handle)}


def append_manifest(row: dict[str, str]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    new_file = not MANIFEST.exists() or MANIFEST.stat().st_size == 0
    with MANIFEST.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    known_urls = read_existing_urls()
    session = requests.Session()
    session.headers.update(HEADERS)
    downloaded = 0

    for source in SOURCES:
        try:
            response = session.get(source, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"[WARN] Could not access {source}: {exc}")
            continue

        for url, title in pdf_links(source, response.text):
            if url in known_urls:
                continue
            # Keep requests polite; this delay is required by the assignment.
            time.sleep(2)
            try:
                pdf_response = session.get(url, timeout=45)
                pdf_response.raise_for_status()
                if not pdf_response.content.startswith(b"%PDF"):
                    print(f"[SKIP] Response is not a PDF: {url}")
                    continue
                filename = Path(urlparse(url).path).name or f"rbi_{downloaded}.pdf"
                target = RAW_DIR / filename
                if target.exists():
                    target = RAW_DIR / f"{downloaded}_{filename}"
                target.write_bytes(pdf_response.content)
                append_manifest({
                    "file": target.name,
                    "title": title,
                    "date": "",
                    "source_url": url,
                    "pages": "",
                    "parser": "",
                })
                known_urls.add(url)
                downloaded += 1
                print(f"[OK] {downloaded}: {target.name}")
            except requests.RequestException as exc:
                print(f"[WARN] PDF download failed: {url} ({exc})")

    print(f"Downloaded this run: {downloaded}")
    print(f"Total PDFs in raw/: {len(list(RAW_DIR.glob('*.pdf')))}")
    print(f"Manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
