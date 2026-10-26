"""Central configuration. Everything is overridable via environment variables / .env."""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    strong_model: str = field(default_factory=lambda: _env("STRONG_MODEL", "claude-sonnet-5-5"))  # planner, SQL, generator, self-check
    cheap_model: str = field(default_factory=lambda: _env("CHEAP_MODEL", "claude-haiku-5-5"))  # grader, rewriter
    max_retries: int = 3  # shared budget for rewrites + regenerations
    max_rows: int = 200  # hard cap on SQL result rows
    rows_in_prompt: int = 50  # rows of a SQL result shown to the LLM
    top_k: int = 5
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(ROOT / "data"))))
    price_in_per_mtok: float = field(default_factory=lambda: float(_env("PRICE_IN_PER_MTOK", "0")))
    price_out_per_mtok: float = field(default_factory=lambda: float(_env("PRICE_OUT_PER_MTOK", "0")))

    @property
    def docs_dir(self) -> Path:
        return self.data_dir / "docs"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "fans.db"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def ledger_path(self) -> Path:
        return self.data_dir / "ingest_ledger.json"


settings = Settings()
