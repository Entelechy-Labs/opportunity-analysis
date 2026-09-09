"""Baseline scaffold integrity test for the opportunity-analysis child repository."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_ENTRY = ROOT / "src" / "agent_opportunity_analysis" / "__init__.py"


def test_pyproject_exists() -> None:
    assert (ROOT / "pyproject.toml").is_file()


def test_package_exists() -> None:
    assert SRC_ENTRY.is_file()
