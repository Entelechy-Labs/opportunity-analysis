"""Typed definition of the opportunity-analysis agent.

The ``AgentConfig`` is the definition-time contract of this agent: mission,
model role, allowed/forbidden tools, output contracts, and the JSON schema
its LLM calls must satisfy. The router validates model output against this
schema (fail closed) before the agent ever sees it.
"""

from __future__ import annotations

from typing import Any

from agent_framework.config import AgentConfig
from linkgen_contracts.llm.models import LLMRole, ResponseFormat

OPPORTUNITY_ANALYSIS_SLUG = "opportunity-analysis"

# JSON schema for the model's structured output. Restricted to the router's
# documented schema subset (type/properties/items) so it fails closed rather
# than being silently ignored. Ids and timestamps are NEVER requested from the
# model — deterministic Python derives them (see agent.py).
OPPORTUNITY_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "angles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "audience_match": {"type": "number"},
                    "novelty": {"type": "number"},
                },
            },
        },
    },
}

_SYSTEM_PROMPT = """You are the opportunity-analysis agent of the LinkGen content system.
Mission: turn completed research into a map of original content opportunities.
Identify high-interest topics, audience problems, misconceptions, common
mistakes, repeated questions, underserved areas, missing opinions, practical
information, service connections, and authority-building opportunities.
Rate audience match and novelty honestly, never above what the research
supports. Do not copy creator wording or distinctive material; every angle
must be a new angle, not a rephrased post.
Respond only with JSON matching the requested schema."""


def build_opportunity_analysis_config() -> AgentConfig:
    """Build the definition-time configuration of the opportunity agent."""
    return AgentConfig(
        slug=OPPORTUNITY_ANALYSIS_SLUG,
        model_role=LLMRole.REASONING,
        output_contracts=[
            "OpportunityMap",
            "OpportunityAngle",
        ],
        system_prompt=_SYSTEM_PROMPT,
        temperature=0.0,
        max_output_tokens=2048,
        max_steps=8,
        tools=["web-search", "browser"],
        forbidden_tools=["publish", "schedule", "linkedin-post"],
        response_format=ResponseFormat.JSON,
        json_schema=OPPORTUNITY_OUTPUT_SCHEMA,
        description="Identify content gaps and opportunities (strategy phase).",
        version="1.0.0",
    )
