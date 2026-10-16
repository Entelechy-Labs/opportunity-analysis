"""Unit tests for opportunity-angle normalization (agent-specific logic).

Generic sanitizer behavior is covered by agent-framework's tests; here we
test how this agent applies it to model-proposed angles.
"""

from __future__ import annotations

import pytest

from agent_framework.errors import AgentOutputError
from agent_opportunity_analysis.agent import (
    MAX_ANGLES,
    dedupe_angles,
    normalize_angles,
)


def test_normalize_angles_filters_and_validates() -> None:
    payload = {"angles": [{"title": "a"}, "junk", 5, {"title": "b"}]}
    assert normalize_angles(payload) == [{"title": "a"}, {"title": "b"}]
    assert normalize_angles({}) == []
    assert normalize_angles({"angles": None}) == []
    with pytest.raises(AgentOutputError):
        normalize_angles({"angles": "nope"})


def test_dedupe_angles_by_normalized_title() -> None:
    candidates = [
        {"title": " Pricing myths ", "description": "d1"},
        {"title": "pricing myths", "description": "duplicate"},
        {"title": "Onboarding gaps", "description": "d2"},
        {"title": "", "description": "dropped — no title"},
        {"title": "   ", "description": "dropped — blank title"},
        {"title": "missing description", "description": "  "},
    ]
    out = dedupe_angles(candidates)
    assert [entry["title"] for entry in out] == [
        " Pricing myths ",
        "Onboarding gaps",
    ]


def test_dedupe_angles_caps_deterministically() -> None:
    candidates = [
        {"title": f"Angle {i}", "description": f"description {i}"}
        for i in range(MAX_ANGLES + 5)
    ]
    out = dedupe_angles(candidates)
    assert len(out) == MAX_ANGLES
    assert out[0]["title"] == "Angle 0"
    assert out[-1]["title"] == f"Angle {MAX_ANGLES - 1}"
