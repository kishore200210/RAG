"""Parse PDFs in data/raw into Markdown files and update manifest metadata."""
from __future__ import annotations

import csv
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PARSED_DIR = ROOT / "data" / "parsed"
MANIFEST = ROOT / "data" / "manifest.csv"
FIELDS = ["file", "title", "date", "source_url", "pages", "parser"]


def parse_pdf(pdf_path: Path) -> tuple[str, int]:
    pages: list[str] = []
    with fitz.open(pdf_path) as document:
        page_count = len(document)
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            pages.append(f"\n\n## Page {page_number}\n\n{text}")
    return f"# {pdf_path.stem}\n" + "".join(pages), page_count


def main() -> None:
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(
        path for path in RAW_DIR.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )
    if not pdfs:
        raise SystemExit(f"No PDFs found in {RAW_DIR}. Download or add PDFs first.")

    manifest_rows: dict[str, dict[str, str]] = {}
    if MANIFEST.exists():
        with MANIFEST.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                manifest_rows[row.get("file", "")] = row

    for pdf_path in pdfs:
        try:
            markdown, page_count = parse_pdf(pdf_path)
            (PARSED_DIR / f"{pdf_path.stem}.md").write_text(markdown, encoding="utf-8")
            row = manifest_rows.get(pdf_path.name, {
                "file": pdf_path.name,
                "title": pdf_path.stem,
                "date": "",
                "source_url": "",
                "pages": "",
                "parser": "",
            })
            row["pages"] = str(page_count)
            row["parser"] = "pymupdf"
            manifest_rows[pdf_path.name] = row
            print(f"[OK] {pdf_path.name}: {page_count} pages")
        except Exception as exc:  # continue so one malformed PDF doesn't stop the corpus
            print(f"[ERROR] {pdf_path.name}: {exc}")

    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows.values())

    print(f"Parsed {len(list(PARSED_DIR.glob('*.md')))} PDFs into {PARSED_DIR}")


if __name__ == "__main__":
    main()
