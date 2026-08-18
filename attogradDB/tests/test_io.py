import pytest
from pypdf import PdfWriter

from attogradDB.io import TextSplitter
from attogradDB.utils import read_pdf


def test_chunks_cover_the_whole_text_with_overlap():
    splitter = TextSplitter(chunk_size=10, chunk_overlap=3)
    splitter.split_text("abcdefghijklmnopqrstuvwxyz")
    docs = splitter.get_docs()

    assert docs[0] == "abcdefghij"
    assert docs[1].startswith("hij"), "chunks should overlap by chunk_overlap characters"
    assert "".join(d[3:] if i else d for i, d in enumerate(docs)) == "abcdefghijklmnopqrstuvwxyz"


def test_split_text_resets_between_calls():
    splitter = TextSplitter(chunk_size=5, chunk_overlap=0)
    splitter.split_text("aaaaa")
    splitter.split_text("bbbbb")
    assert splitter.get_docs() == ["bbbbb"]


@pytest.mark.parametrize(
    "chunk_size,chunk_overlap",
    [(100, 100), (100, 150), (0, 0), (-1, 0)],
)
def test_rejects_configurations_that_cannot_advance(chunk_size, chunk_overlap):
    """An overlap at or above chunk_size makes split_text loop forever."""
    with pytest.raises(ValueError):
        TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def test_read_pdf_extracts_text():
    text = read_pdf("sample_data/leclerc_sample.pdf")
    assert isinstance(text, str) and text.strip()


def test_read_pdf_tolerates_pages_without_a_text_layer(tmp_path):
    """Scanned/image-only pages yield no text; that must not abort the whole read."""
    path = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with open(path, "wb") as f:
        writer.write(f)

    assert read_pdf(str(path)) == ""


def test_read_pdf_missing_file():
    with pytest.raises(FileNotFoundError):
        read_pdf("does_not_exist.pdf")
