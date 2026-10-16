"""Determinism and id-stability regression tests (golden path).

The same input, clock, and router must produce identical artifacts; different
run ids must produce different derived ids (no cross-run bleed).
"""

from __future__ import annotations

from agent_framework.ids import deterministic_id
from agent_opportunity_analysis.agent import OpportunityAnalysisAgent
from linkgen_contracts.common.ids import AngleId, RunId, TaskId

from conftest import RUN_ID, TASK_ID, make_context, make_input


def _run_agent(payload: dict, *, run_id: str = RUN_ID) -> dict:
    agent = OpportunityAnalysisAgent(context=make_context())
    return agent.process(
        run_id=RunId(run_id), task_id=TaskId(TASK_ID), input_data=payload
    )


def test_same_input_same_output() -> None:
    first = _run_agent(make_input())
    second = _run_agent(make_input())
    assert first == second


def test_angle_ids_are_derived_deterministically() -> None:
    output = _run_agent(make_input())
    for angle in output["opportunities"]:
        expected = deterministic_id(AngleId, RUN_ID, "angle", angle["title"].lower())
        assert angle["angle_id"] == expected


def test_different_run_id_yields_different_ids() -> None:
    other_run = "run_" + "e4" * 16
    base = _run_agent(make_input())
    other = _run_agent(make_input(run_id=other_run), run_id=other_run)
    assert base["opportunity_map"]["opportunity_id"] != other["opportunity_map"][
        "opportunity_id"
    ]
    base_angles = {a["angle_id"] for a in base["opportunities"]}
    other_angles = {a["angle_id"] for a in other["opportunities"]}
    if base_angles and other_angles:
        assert base_angles != other_angles


def test_map_id_depends_only_on_run_and_request() -> None:
    # opportunity_id derives from (run_id, request_id) — not from the focus
    # text — so a re-run under the same run/request is idempotent at the
    # artifact level.
    first = _run_agent(make_input())
    second = _run_agent(
        make_input(request=dict(make_input()["request"], focus="different focus"))
    )
    assert (
        first["opportunity_map"]["opportunity_id"]
        == second["opportunity_map"]["opportunity_id"]
    )
