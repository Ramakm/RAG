from pathlib import Path

from agentic_rag import vectorstore
from agentic_rag.ingestion import metadata
from agentic_rag.ingestion.chunking import chunk_sections, split_text
from agentic_rag.ingestion.parsers import Section


def test_metadata_extraction():
    m = metadata.extract("QC Report B-2025-07-A for CF-1200, VoltCap lot CAP-7781 on 2025-07-12")
    assert m == {"batch_id": "B-2025-07-A", "model": "CF-1200", "lot_id": "CAP-7781", "doc_date": "2025-07-12", "supplier": "VoltCap"}


def test_doc_type_from_folder():
    assert metadata.infer_doc_type(Path("data/docs/qc_reports/x.md")) == "qc_report"
    assert metadata.infer_doc_type(Path("elsewhere/x.md")) == "other"


def test_split_text_respects_max_and_overlap():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(10))
    chunks = split_text(text, max_chars=700, overlap=100)
    assert len(chunks) > 1
    assert all(len(c) <= 700 + 100 + 10 for c in chunks)


def test_tables_stay_whole_and_tickets_stay_single():
    table = Section(text="a | b\n" * 100, is_table=True)
    assert len(chunk_sections([table], "supplier_spec")) == 1
    rows = [Section(text=f"ticket {i}: noisy") for i in range(3)]
    assert len(chunk_sections(rows, "service_ticket")) == 3


def test_ingest_indexes_everything_and_is_incremental(seeded):
    from agentic_rag.ingestion.pipeline import ingest

    assert seeded["chunks"] > 100 and seeded["added"] > 50
    again = ingest()
    assert again["skipped"] == seeded["added"] + seeded.get("updated", 0) and again["added"] == 0


def test_excel_rows_carry_their_own_metadata(seeded):
    hits = vectorstore.search("buzzing noise motor housing", 5, {"batch_id": "B-2025-08-E"})
    assert hits and all(meta["batch_id"] == "B-2025-08-E" for _, meta in hits)


def test_docx_table_is_searchable(seeded):
    docs = [d for d, _ in vectorstore.search("VoltCap rated temperature", 5, {"doc_type": "supplier_spec"})]
    assert any("Rated temperature | 85C" in d for d in docs)


def test_changed_file_is_reindexed_and_deleted_file_removed(seeded):
    from agentic_rag.config import settings
    from agentic_rag.ingestion.pipeline import ingest

    p = settings.docs_dir / "manuals" / "EF-150_manual.md"
    p.write_text(p.read_text() + "\n\n## Troubleshooting: smell\nUnplug the fan if it smells like burning.\n")
    assert ingest()["updated"] == 1
    assert any("burning" in d for d, _ in vectorstore.search("smells like burning", 3, {"doc_type": "manual"}))
    p.unlink()
    assert ingest()["removed"] == 1
