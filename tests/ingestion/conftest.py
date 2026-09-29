from pathlib import Path

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def load_fixture(name: str) -> str:
    return (FIXTURES_DIR / "synthetic" / name).read_text(encoding="utf-8")
