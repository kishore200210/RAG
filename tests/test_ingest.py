from src.ingest import parse_pdf


def test_parse_pdf_returns_page_count_and_page_marker(tmp_path):
    import fitz

    pdf_path = tmp_path / "sample.pdf"
    with fitz.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "RBI sample regulation")
        document.save(pdf_path)

    markdown, page_count = parse_pdf(pdf_path)
    assert page_count == 1
    assert "## Page 1" in markdown
    assert "RBI sample regulation" in markdown
