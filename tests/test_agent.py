"""Agent-level tests: normal cases, edge cases, and malformed inputs."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agent_framework.errors import AgentInputError
from agent_opportunity_analysis.agent import (
    OpportunityAnalysisAgent,
    baseline_angle,
    build_angles,
    build_opportunity_map,
    build_prompt,
    validate_input,
)
from agent_opportunity_analysis.config import OPPORTUNITY_ANALYSIS_SLUG
from linkgen_contracts.errors.envelope import ErrorCategory
from linkgen_contracts.research.models import (
    CreatorProfile,
    MarketSignal,
    OpportunityAngle,
    OpportunityMap,
    ResearchRequest,
)
from linkgen_contracts.workflow.models import TaskStatus

from conftest import (
    RUN_ID,
    TASK_ID,
    make_creator,
    make_input,
    make_request,
    make_signal,
)

FIXED_AT = datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc)

CANDIDATES = [
    {
        "title": "Onboarding checklists that stall",
        "description": "Teams copy checklist templates without fitting them.",
        "audience_match": 0.77,
        "novelty": 0.61,
    },
    {
        "title": "Onboarding checklists that stall",  # duplicate title
        "description": "Duplicate entry — dropped by dedupe.",
    },
    {"title": "", "description": "no title — dropped"},
    {"title": "missing description", "description": "   "},
]


class TestInputValidation:
    def test_valid_input_round_trips(self) -> None:
        request, creators, signals = validate_input(make_input())
        assert isinstance(request, ResearchRequest)
        assert len(creators) == 1 and isinstance(creators[0], CreatorProfile)
        assert len(signals) == 1 and isinstance(signals[0], MarketSignal)

    def test_minimal_input_without_optional_artifacts(self) -> None:
        request, creators, signals = validate_input({"request": make_request()})
        assert creators == [] and signals == []

    def test_non_dict_input_rejected(self) -> None:
        with pytest.raises(AgentInputError):
            validate_input("not a dict")

    def test_missing_request_key_rejected(self) -> None:
        with pytest.raises(AgentInputError):
            validate_input({"creators": []})

    def test_empty_focus_rejected(self) -> None:
        with pytest.raises(AgentInputError):
            validate_input(make_input(request=make_request(focus="   ")))

    def test_wrong_types_rejected(self) -> None:
        with pytest.raises(AgentInputError):
            validate_input(make_input(request=make_request(focus=123)))

    def test_invalid_creator_rejected(self) -> None:
        with pytest.raises(AgentInputError):
            validate_input(make_input(creators=[{"name": ""}]))

    def test_invalid_signal_list_rejected(self) -> None:
        with pytest.raises(AgentInputError):
            validate_input(make_input(signals="not a list"))


class TestNormalRun:
    def test_process_produces_contract_valid_output(
        self, agent: OpportunityAnalysisAgent, run_id, task_id
    ) -> None:
        output = agent.process(
            run_id=run_id, task_id=task_id, input_data=make_input()
        )
        assert set(output) == {"opportunity_map", "opportunities"}
        opportunity_map = OpportunityMap.model_validate(output["opportunity_map"])
        angles = [
            OpportunityAngle.model_validate(item) for item in output["opportunities"]
        ]
        assert opportunity_map.run_id == run_id
        assert opportunity_map.opportunities == angles
        assert opportunity_map.signals  # research signals carried over
        assert opportunity_map.top_opportunity == angles[0].title
        for angle in angles:
            assert str(angle.angle_id).startswith("angle_")
            assert angle.title and angle.description
            assert 0.0 <= angle.audience_match <= 1.0
            assert 0.0 <= angle.novelty <= 1.0

    def test_map_is_always_populated(
        self, agent: OpportunityAnalysisAgent, run_id, task_id
    ) -> None:
        # Even when the model yields no usable angle, the strategy phase must
        # receive a focus-derived baseline angle (never an empty map).
        output = agent.process(
            run_id=run_id, task_id=task_id, input_data=make_input()
        )
        opportunity_map = OpportunityMap.model_validate(output["opportunity_map"])
        assert opportunity_map.opportunities
        assert opportunity_map.top_opportunity

    def test_run_reports_success(
        self, agent: OpportunityAnalysisAgent, run_id, task_id
    ) -> None:
        result = agent.run(
            run_id=run_id, task_id=task_id, input_data=make_input()
        )
        assert result.status is TaskStatus.SUCCEEDED
        assert result.errors == []

    def test_config_declares_expected_boundaries(
        self, agent: OpportunityAnalysisAgent
    ) -> None:
        assert agent.config.slug == OPPORTUNITY_ANALYSIS_SLUG
        assert "publish" in agent.config.forbidden_tools
        assert "schedule" in agent.config.forbidden_tools
        assert agent.config.response_format.value == "json"


class TestRunFailures:
    def test_malformed_input_maps_to_failed_result(
        self, agent: OpportunityAnalysisAgent, run_id, task_id
    ) -> None:
        result = agent.run(run_id=run_id, task_id=task_id, input_data="junk")
        assert result.status is TaskStatus.FAILED
        assert result.errors[0].category is ErrorCategory.VALIDATION

    def test_missing_request_maps_to_failed_result(
        self, agent: OpportunityAnalysisAgent, run_id, task_id
    ) -> None:
        result = agent.run(run_id=run_id, task_id=task_id, input_data={})
        assert result.status is TaskStatus.FAILED
        assert result.errors[0].code == "agent_input_invalid"


class TestArtifactBuilders:
    def test_dedupe_keeps_first_occurrence_and_drops_unusable(self) -> None:
        from agent_opportunity_analysis.agent import dedupe_angles

        kept = dedupe_angles(CANDIDATES)
        assert len(kept) == 1
        assert kept[0]["title"] == "Onboarding checklists that stall"

    def test_build_angles_ids_and_clamped_scores(self, run_id) -> None:
        angles = build_angles(run_id=run_id, candidates=CANDIDATES[:1])
        assert len(angles) == 1
        angle = angles[0]
        assert angle.audience_match == 0.77
        assert angle.novelty == 0.61
        # Ids derive from (run, title) — stable, never random.
        again = build_angles(run_id=run_id, candidates=CANDIDATES[:1])
        assert again[0].angle_id == angle.angle_id

    def test_out_of_range_scores_are_clamped(self, run_id) -> None:
        angles = build_angles(
            run_id=run_id,
            candidates=[
                {"title": "Wide", "description": "d", "audience_match": 42.0,
                 "novelty": -3.0}
            ],
        )
        assert angles[0].audience_match == 1.0
        assert angles[0].novelty == 0.0

    def test_baseline_angle_derives_from_focus(self, run_id) -> None:
        request = ResearchRequest.model_validate(make_request())
        angle = baseline_angle(run_id=run_id, request=request)
        assert request.focus in angle.title
        assert 0.0 <= angle.audience_match <= 1.0
        assert angle.angle_id == baseline_angle(
            run_id=run_id, request=request
        ).angle_id

    def test_map_top_opportunity_and_signal_cap(self, run_id) -> None:
        request = ResearchRequest.model_validate(make_request())
        signals = [
            MarketSignal.model_validate(make_signal()) for _ in range(30)
        ]
        angles = build_angles(run_id=run_id, candidates=CANDIDATES[:1])
        opportunity_map = build_opportunity_map(
            run_id=run_id,
            request=request,
            angles=angles,
            signals=signals,
            generated_at=FIXED_AT,
        )
        assert opportunity_map.top_opportunity == angles[0].title
        assert len(opportunity_map.signals) == 20  # bounded
        assert opportunity_map.opportunity_id == build_opportunity_map(
            run_id=run_id,
            request=request,
            angles=angles,
            signals=signals,
            generated_at=FIXED_AT,
        ).opportunity_id

    def test_prompt_contains_focus_and_ground_rules(self) -> None:
        request = ResearchRequest.model_validate(make_request())
        prompt = build_prompt(request, [], [])
        assert request.focus in prompt
        assert "never copy creator wording" in prompt
        assert "Return JSON with an 'angles' array." in prompt
