"""Reading PDF, DOCX and TXT files: text layout, page numbers, headings, failures."""
import io

import pytest

from jargonaut.ingest import IngestError, detect_heading, read_document

BODY = ("Either party may terminate this Agreement for convenience upon thirty (30) days prior written notice "
        "to the other party, provided that all fees accrued up to the date of termination remain payable. ")


def make_pdf(pages):
    """pages: list of lists of (text, bold) lines; adds a running header and a page-number footer."""
    import pymupdf

    pdf = pymupdf.open()
    for n, lines in enumerate(pages, 1):
        page = pdf.new_page(width=612, height=792)
        page.insert_text((72, 30), "ACME CORP - CONFIDENTIAL", fontsize=9)
        y = 90
        for text, bold in lines:
            page.insert_textbox(pymupdf.Rect(72, y, 540, y + 200), text, fontsize=11,
                                fontname="hebo" if bold else "helv")
            y += 16 * (1 + len(text) // 85) + 12
        page.insert_text((300, 770), f"Page {n} of {len(pages)}", fontsize=9)
    return pdf.tobytes()


def test_pdf_pages_headers_and_headings():
    data = make_pdf([
        [("MASTER SERVICES AGREEMENT", True), ("1. DEFINITIONS", True), ("In this Agreement the following terms apply.", False)],
        [("9. Term and Termination", True), ("9.2 Termination for Convenience. " + BODY, False)],
        [("10. INSURANCE", True), ("The Supplier shall maintain general liability insurance of $1,000,000.", False)],
    ])
    doc = read_document(data, "msa.pdf")
    assert doc.kind == "pdf" and doc.n_pages == 3
    assert "CONFIDENTIAL" not in doc.text and "Page 2 of 3" not in doc.text     # running header/footer removed
    i = doc.text.index("Either party may terminate")
    assert doc.page_at(i) == 2
    assert doc.pages(i, doc.text.index("remain payable")) == [2]
    assert doc.page_at(doc.text.index("general liability insurance")) == 3
    section = [h.text for h in doc.section(i)]
    assert section[0].startswith("9. Term and Termination") and section[-1].startswith("9.2 Termination for Convenience")
    assert [h.text for h in doc.section(doc.text.index("general liability"))][-1] == "10. INSURANCE"
    # wrapped lines are joined back into one paragraph per line, as in CUAD's text
    assert "thirty (30) days prior written notice to the other party" in doc.text


def test_pdf_paragraph_split_by_page_break_is_rejoined():
    first = "The Licensee shall not assign this Agreement or any of its rights or obligations hereunder"
    data = make_pdf([[(first, False)], [("without the prior written consent of the Licensor.", False)]])
    doc = read_document(data, "split.pdf")
    joined = first + " without the prior written consent of the Licensor."
    assert joined in doc.text
    assert doc.page_at(doc.text.index("without the prior")) == 2 and doc.page_at(doc.text.index("Licensee")) == 1


def test_pdf_heading_wrapped_over_two_lines_stays_whole():
    data = make_pdf([[("SECTION 11. Survival of Agreements, Representations and\nIndemnities", True),
                      ("All representations shall survive the closing.", False)]])
    doc = read_document(data, "wrap.pdf")
    assert [h.text for h in doc.section(doc.text.index("All representations"))] == [
        "SECTION 11. Survival of Agreements, Representations and Indemnities"]


def test_pdf_page_numbers_between_paragraphs_are_removed():
    """A web page printed to PDF (most EDGAR filings): several original pages per PDF page,
    numbered in mid-page, sometimes in the middle of a sentence."""
    import pymupdf

    pdf = pymupdf.open()
    page = pdf.new_page(width=612, height=792)
    for y, x, text in [
        (72, 72, "The Licensee shall keep complete and accurate records of all sales of the"),
        (88, 72, "Products and shall permit the Licensor to inspect them"),
        (130, 300, "1"),
        (146, 72, "Source: ACME CORP, 8-K, 3/1/2010"),
        (188, 72, "on reasonable notice during normal business hours."),
        (220, 72, "The annual minimum fee is set out below."),
        (252, 300, "250"),
        (284, 72, "The Licensee shall not sell the Products outside the Territory."),
        (326, 300, "2"),
        (342, 72, "Source: ACME CORP, 8-K, 3/1/2010"),
        (384, 72, "Either party may terminate this Agreement on ninety (90) days notice."),
    ]:
        page.insert_text((x, y), text, fontsize=11)
    doc = read_document(pdf.tobytes(), "edgar.pdf")
    lines = doc.text.split("\n")
    assert "1" not in lines and "2" not in lines                  # a run of page numbers
    assert "Source:" not in doc.text                               # the footer that recurs next to them
    assert "250" in lines                                          # a lone number, not in a run, is kept
    assert "inspect them on reasonable notice during normal business hours." in doc.text   # sentence rejoined
    assert doc.n_pages == 1


def test_pdf_clause_number_set_apart_from_its_title():
    import pymupdf

    pdf = pymupdf.open()
    page = pdf.new_page(width=612, height=792)
    for y, x, text, bold in [
        (72, 108, "This agreement shall be governed by the Laws of England.", False),
        (100, 72, "11.", True), (100, 108, "INTELLECTUAL PROPERTY", True),
        (118, 108, "The Manufacturer owns the intellectual property in the Products:", False),
        (136, 72, "(a)", False), (136, 108, "including all trade marks; and", False),
    ]:
        page.insert_text((x, y), text, fontsize=11, fontname="hebo" if bold else "helv")
    doc = read_document(pdf.tobytes(), "hanging.pdf")
    lines = doc.text.split("\n")
    assert lines[0] == "This agreement shall be governed by the Laws of England."
    assert [h.text for h in doc.section(doc.text.index("The Manufacturer"))] == ["11. INTELLECTUAL PROPERTY"]
    assert "(a) including all trade marks; and" in lines


def test_pdf_odd_characters_are_normalised():
    from jargonaut.ingest import _clean

    assert _clean("the Licensor; and the  Licensee­") == "the Licensor; and the Licensee-"
    assert _clean("ﬁnal notiﬁcation​") == "final notification"


def test_scanned_pdf_is_rejected():
    import pymupdf

    pdf = pymupdf.open()
    pdf.new_page()
    with pytest.raises(IngestError, match="no text layer"):
        read_document(pdf.tobytes(), "scan.pdf")


def make_docx(rendered_page_breaks=True):
    import docx
    from docx.oxml import OxmlElement

    d = docx.Document()
    d.add_heading("Distribution Agreement", level=0)
    d.add_heading("Term and Termination", level=1)
    d.add_paragraph("This Agreement shall expire on December 31, 2027.", style="List Number")
    p = d.add_paragraph(style="List Number")
    run = p.add_run("Termination for Convenience.")
    run.bold = True
    p.add_run(" " + BODY)
    d.add_heading("Insurance", level=1)
    p = d.add_paragraph()
    run = p.add_run("The Distributor shall maintain product liability insurance.")
    if rendered_page_breaks:     # what Word writes where it last broke the page
        run._r.insert(0, OxmlElement("w:lastRenderedPageBreak"))
    table = d.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Coverage"
    table.rows[0].cells[1].text = "$2,000,000 per occurrence"
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def test_docx_numbering_headings_pages_tables():
    doc = read_document(make_docx(), "agreement.docx")
    assert doc.kind == "docx"
    # Word's automatic numbers are rendered into the text
    assert "1. This Agreement shall expire on December 31, 2027." in doc.text
    assert "2. Termination for Convenience. Either party may terminate" in doc.text
    assert "Coverage | $2,000,000 per occurrence" in doc.text
    i = doc.text.index("Either party may terminate")
    path = [h.text for h in doc.section(i)]
    assert path[0] == "Distribution Agreement" and "Term and Termination" in path
    assert path[-1] == "2. Termination for Convenience"           # bold run-in title
    assert doc.n_pages == 2 and doc.page_at(i) == 1
    assert doc.page_at(doc.text.index("product liability insurance")) == 2
    assert [h.text for h in doc.section(doc.text.index("product liability"))][-1] == "Insurance"


def test_docx_without_saved_layout_has_no_pages():
    doc = read_document(make_docx(rendered_page_breaks=False), "generated.docx")
    assert doc.page_starts is None and doc.pages(0, 10) is None
    assert any("page numbers are not available" in w for w in doc.warnings)


def test_txt_form_feeds_are_pages():
    doc = read_document("ARTICLE 1 TERM\nThis Agreement lasts two years.\fARTICLE 2 LAW\nNew York law applies.".encode(), "c.txt")
    assert doc.n_pages == 2 and doc.page_at(doc.text.index("New York")) == 2
    assert [h.text for h in doc.headings] == ["ARTICLE 1 TERM", "ARTICLE 2 LAW"]


def test_unsupported_files():
    with pytest.raises(IngestError, match=".doc files"):
        read_document(b"xx", "old.doc")
    with pytest.raises(IngestError, match="Unsupported"):
        read_document(b"xx", "image.png")


@pytest.mark.parametrize("text, expected", [
    ("ARTICLE 9 TERM AND TERMINATION", (1, "ARTICLE 9 TERM AND TERMINATION")),
    ("Section 9.2 Termination", (2, "Section 9.2 Termination")),
    ("9. Term and Termination", (1, "9. Term and Termination")),
    ("9.2 TERMINATION", (2, "9.2 TERMINATION")),
    ("9.2 Termination for Convenience. Either party may terminate this Agreement.", (2, "9.2 Termination for Convenience")),
    ("GOVERNING LAW", (1, "GOVERNING LAW")),
    ("EXHIBIT 10.13", (1, "EXHIBIT 10.13")),
    ("SECTION 12. Termination. Capital Resources may terminate this Agreement by giving the notice indicated below "
     "in this Section at any time.", (1, "SECTION 12. Termination")),
    ("Exhibit A attached hereto, describes by way of example the products.", None),
    ("Section 5 of this Agreement shall survive", None),
    ("2. The Company shall pay the fees.", None),
    ("The Company shall pay all fees within thirty days.", None),
    ("NOW, THEREFORE,", None),
])
def test_detect_heading(text, expected):
    assert detect_heading(text) == expected
