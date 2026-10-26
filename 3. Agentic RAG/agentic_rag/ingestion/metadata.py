"""Metadata extraction. Regex first (IDs are deterministic); adapt the patterns to your own ID formats."""
import re
from pathlib import Path

BATCH_RE = re.compile(r"\bB-\d{4}-\d{2}-[A-Z]\b")
MODEL_RE = re.compile(r"\b(?:CF|TF|EF)-\d{3,4}\b")
LOT_RE = re.compile(r"\b(?:CAP|KX|ZN)-\d{3,5}\b")
DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
KNOWN_SUPPLIERS = ["VoltCap", "Kapex", "Zenith"]  # in production: load from a suppliers table

FOLDER_TO_TYPE = {
    "qc_reports": "qc_report",
    "supplier_specs": "supplier_spec",
    "manuals": "manual",
    "service_tickets": "service_ticket",
}


def infer_doc_type(path: Path) -> str:
    for part in path.parts:
        if part in FOLDER_TO_TYPE:
            return FOLDER_TO_TYPE[part]
    return "other"


def extract(text: str) -> dict:
    """First match of each ID type found in `text`; missing keys are omitted."""
    found = {}
    if m := BATCH_RE.search(text):
        found["batch_id"] = m.group()
    if m := MODEL_RE.search(text):
        found["model"] = m.group()
    if m := LOT_RE.search(text):
        found["lot_id"] = m.group()
    if m := DATE_RE.search(text):
        found["doc_date"] = m.group(1)
    for s in KNOWN_SUPPLIERS:
        if re.search(rf"\b{s}\b", text, re.I):
            found["supplier"] = s
            break
    return found
