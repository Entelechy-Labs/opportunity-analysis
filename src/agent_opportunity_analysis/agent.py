"""Opportunity-analysis agent — Phase 09 strategy system.

Consumes the research phase outputs and produces a contract-conforming
``OpportunityMap``:

- ``OpportunityAngle[]`` (one per distinct content angle)
- ``MarketSignal[]`` carried over from the research phase (validated, capped)

Design rules (shared with the Phase 08 research agents):
- The LLM is used only for content judgment; ids, timestamps, dedupe, capping,
  and confidence bounds are deterministic Python.
- Model output is untrusted: it arrives router-validated as JSON, and every
  field is sanitized before entering a contract model.
- Ids are derived deterministically (uuid5) so identical inputs produce
  identical artifacts (idempotent replay, golden tests, checkpoint recovery).
- No network, clock, or randomness: time comes from the injected context
  clock; failures come from the router as duck-typed-normalized errors.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from linkgen_contracts.common.ids import AngleId, ResultId, RunId, TaskId
from linkgen_contracts.errors.envelope import ErrorCategory
from linkgen_contracts.research.models import (
    CreatorProfile,
    MarketSignal,
    OpportunityAngle,
    OpportunityMap,
    ResearchRequest,
)
from pydantic import ValidationError

from agent_framework.base import AgentBase
from agent_framework.config import AgentConfig
from agent_framework.context import AgentContext
from agent_framework.errors import (
    AgentError,
    AgentInputError,
    AgentOutputError,
)
from agent_framework.ids import deterministic_id
from agent_framework.sanitize import (
    bounded_str_list,
    clean_text,
    normalize_key,
    safe_confidence,
    stable_probability,
)
from agent_framework.structured import extract_json_dict

from agent_opportunity_analysis.config import (
    OPPORTUNITY_OUTPUT_SCHEMA,
    build_opportunity_analysis_config,
)

# Bounded resource usage (AGENTS rule 11): the source spec targets a focused
# 30-day strategy, so the caps below are generous hard limits, never unbounded
# growth.
MAX_ANGLES = 20
MAX_SIGNALS = 20

__all__ = [
    "OpportunityAnalysisAgent",
    "baseline_angle",
    "build_angles",
    "build_opportunity_map",
    "build_prompt",
    "dedupe_angles",
    "normalize_angles",
    "validate_input",
    "normalize_router_failure",
]


# ---------------------------------------------------------------------------
# Untrusted-input normalization (agent-specific; sanitizers are shared).
# ---------------------------------------------------------------------------

def normalize_angles(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Filter the untrusted ``angles`` array down to dict entries."""
    raw = payload.get("angles")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise AgentOutputError("model output field 'angles' must be an array")
    return [entry for entry in raw if isinstance(entry, dict)]


def dedupe_angles(
    candidates: list[dict[str, Any]], *, cap: int = MAX_ANGLES
) -> list[dict[str, Any]]:
    """Dedupe by normalized title; drop unusable entries; cap deterministically.

    An angle needs both a title and a description to be actionable, so
    entries missing either are dropped. Duplicates keep the first occurrence
    and the cap keeps the first ``cap`` entries: the result is a
    deterministic function of the input order.
    """
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for entry in candidates:
        title = clean_text(entry.get("title"))
        description = clean_text(entry.get("description"))
        key = normalize_key(title)
        if not key or not description:
            continue  # not actionable — skip deterministically
        if key in seen:
            continue
        seen.add(key)
        out.append(entry)
        if len(out) >= cap:
            break
    return out


def _validate_model_list(
    raw: Any, model: type, label: str
) -> list[Any]:
    """Validate an optional list of already-contracted input artifacts."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise AgentInputError(f"opportunity-analysis input '{label}' must be a list")
    validated: list[Any] = []
    for index, item in enumerate(raw):
        try:
            validated.append(model.model_validate(item))
        except ValidationError as exc:
            first = exc.errors()[0]
            location = ".".join(str(part) for part in first.get("loc", ()))
            raise AgentInputError(
                f"{label}[{index}] failed contract validation at '{location}': "
                f"{first.get('msg', 'invalid')}"
            ) from exc
    return validated


def validate_input(
    input_data: Any,
) -> tuple[ResearchRequest, list[CreatorProfile], list[MarketSignal]]:
    """Validate the agent input against the research-phase contracts.

    Input shape::

        {"request": ResearchRequest,                 # required
         "creators": [CreatorProfile, ...],           # optional
         "signals": [MarketSignal, ...]}              # optional
    """
    if not isinstance(input_data, dict):
        raise AgentInputError(
            "opportunity-analysis input must be a dict with a 'request' key"
        )
    raw_request = input_data.get("request")
    if raw_request is None:
        raise AgentInputError(
            "opportunity-analysis input is missing the 'request' object"
        )
    try:
        request = ResearchRequest.model_validate(raw_request)
    except ValidationError as exc:
        first = exc.errors()[0]
        location = ".".join(str(part) for part in first.get("loc", ()))
        raise AgentInputError(
            f"research request failed contract validation at '{location}': "
            f"{first.get('msg', 'invalid')}"
        ) from exc
    creators = _validate_model_list(input_data.get("creators"), CreatorProfile, "creators")
    signals = _validate_model_list(input_data.get("signals"), MarketSignal, "signals")
    return request, creators, signals


class _RouterFailure(AgentError):
    """Agent-level wrapper for a router failure (category/code preserved)."""


def normalize_router_failure(exc: Exception) -> AgentError:
    """Normalize a router failure without importing provider internals.

    Router errors carry ``category``/``code``/``retryable``; they are preserved
    by duck-typing so ``TaskResult`` mapping keeps retry semantics honest.
    """
    failure = _RouterFailure(f"router call failed: {getattr(exc, 'message', None) or exc}")
    raw_category = getattr(exc, "category", ErrorCategory.INTERNAL)
    failure.category = raw_category if isinstance(raw_category, ErrorCategory) else ErrorCategory.INTERNAL
    failure.code = str(getattr(exc, "code", "router_failure"))
    failure.retryable = bool(getattr(exc, "retryable", False))
    return failure


def build_prompt(
    request: ResearchRequest,
    creators: list[CreatorProfile],
    signals: list[MarketSignal],
) -> str:
    """Deterministic user prompt for the opportunity-analysis LLM call."""
    lines = [
        "Map content opportunities for a LinkedIn content system.",
        f"Focus: {request.focus}",
    ]
    if request.objectives:
        lines.append("Objectives: " + "; ".join(request.objectives))
    if creators:
        lines.append(
            "Creators already researched: "
            + ", ".join(f"{c.name} ({c.handle})" for c in creators[:10])
        )
    if signals:
        lines.append("Market signals observed:")
        lines.extend(f"- {s.description}" for s in signals[:10])
    lines.append(
        "Identify high-interest topics, audience problems, misconceptions, "
        "mistakes, repeated questions, underserved areas, missing opinions, "
        "practical information, service connections, and authority-building "
        "opportunities. Produce original angles: never copy creator wording "
        "or distinctive material. Rate audience_match and novelty honestly "
        "within [0, 1]."
    )
    lines.append("Return JSON with an 'angles' array.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Artifact builders (pure functions of their inputs).
# ---------------------------------------------------------------------------

def build_angles(
    *, run_id: RunId, candidates: list[dict[str, Any]]
) -> list[OpportunityAngle]:
    """Map untrusted, deduped angle entries into contract-valid angles.

    ``candidates`` must already be deduped (``dedupe_angles``).
    """
    angles: list[OpportunityAngle] = []
    for entry in candidates:
        title = clean_text(entry.get("title"))
        description = clean_text(entry.get("description"))
        if not title or not description:
            continue
        angles.append(
            OpportunityAngle(
                angle_id=deterministic_id(
                    AngleId, str(run_id), "angle", normalize_key(title)
                ),
                title=title,
                description=description,
                audience_match=safe_confidence(entry.get("audience_match")),
                novelty=safe_confidence(entry.get("novelty")),
            )
        )
    return angles


def baseline_angle(*, run_id: RunId, request: ResearchRequest) -> OpportunityAngle:
    """The deterministic angle derived from the request focus itself.

    Used only when the model produced no usable angle, so the strategy phase
    always has at least one *input-derived* (never invented) angle to work
    with. Scores are stable probabilities of the run/request, not claims.
    """
    title = f"Focus area: {clean_text(request.focus) or 'unspecified'}"
    description = (
        f"Content angle anchored on the research focus "
        f"'{clean_text(request.focus) or 'unspecified'}' for run {run_id}."
    )
    return OpportunityAngle(
        angle_id=deterministic_id(
            AngleId, str(run_id), "angle", normalize_key(title)
        ),
        title=title,
        description=description,
        audience_match=stable_probability(
            "baseline-audience", str(run_id), str(request.request_id)
        ),
        novelty=stable_probability(
            "baseline-novelty", str(run_id), str(request.request_id)
        ),
    )


def build_opportunity_map(
    *,
    run_id: RunId,
    request: ResearchRequest,
    angles: list[OpportunityAngle],
    signals: list[MarketSignal],
    generated_at: datetime,
) -> OpportunityMap:
    """Build the run's ``OpportunityMap``."""
    return OpportunityMap(
        opportunity_id=deterministic_id(
            ResultId, str(run_id), "opportunity-map", str(request.request_id)
        ),
        run_id=run_id,
        generated_at=generated_at,
        signals=signals[:MAX_SIGNALS],
        opportunities=angles,
        top_opportunity=angles[0].title if angles else None,
        confidence=stable_probability(
            "opportunity-map", str(run_id), str(request.request_id)
        ),
    )


# ---------------------------------------------------------------------------
# The agent.
# ---------------------------------------------------------------------------

class OpportunityAnalysisAgent(AgentBase):
    """Identify content gaps and opportunities (strategy phase, Phase 09).

    Input (validated against research-phase contracts)::

        {"request": ResearchRequest, "creators"?: [CreatorProfile],
         "signals"?: [MarketSignal]}

    Output::

        {"opportunity_map": OpportunityMap, "opportunities": [OpportunityAngle]}
    """

    def __init__(
        self, context: AgentContext, config: AgentConfig | None = None
    ) -> None:
        super().__init__(
            config if config is not None else build_opportunity_analysis_config(),
            context,
        )

    def process(
        self,
        *,
        run_id: RunId,
        task_id: TaskId,
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        request, creators, signals = validate_input(input_data)
        payload = self._gather(
            run_id=run_id, task_id=task_id,
            request=request, creators=creators, signals=signals,
        )

        angles = build_angles(
            run_id=run_id, candidates=dedupe_angles(normalize_angles(payload))
        )
        if not angles:
            angles = [baseline_angle(run_id=run_id, request=request)]

        opportunity_map = build_opportunity_map(
            run_id=run_id,
            request=request,
            angles=angles,
            signals=signals,
            generated_at=self.context.now(),
        )
        return {
            "opportunity_map": opportunity_map.model_dump(mode="json"),
            "opportunities": [angle.model_dump(mode="json") for angle in angles],
        }

    def _gather(
        self,
        *,
        run_id: RunId,
        task_id: TaskId,
        request: ResearchRequest,
        creators: list[CreatorProfile],
        signals: list[MarketSignal],
    ) -> dict[str, Any]:
        """Call the router for structured opportunity angles (single LLM hop).

        Fault markers in the prompt (mock mode) or a failing adapter surface
        as normalized agent errors; they never escape as raw provider errors.
        """
        prompt = build_prompt(request, creators, signals)
        llm_request = self.build_llm_request(
            run_id=run_id,
            task_id=task_id,
            prompt=prompt,
            json_schema=OPPORTUNITY_OUTPUT_SCHEMA,
        )
        try:
            result = self.context.router.complete_json(
                llm_request, OPPORTUNITY_OUTPUT_SCHEMA
            )
        except AgentError:
            raise
        except Exception as exc:  # noqa: BLE001 — normalized below, never swallowed
            raise normalize_router_failure(exc) from exc
        return extract_json_dict(result)
