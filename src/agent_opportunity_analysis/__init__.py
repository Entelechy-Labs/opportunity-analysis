"""opportunity-analysis — maps content gaps and opportunities (Phase 09).

Strategy-phase agent of the LinkGen content-intelligence system. Consumes the
research phase outputs (``ResearchRequest`` plus optional ``CreatorProfile[]``
and ``MarketSignal[]``) and produces a contract-conforming ``OpportunityMap``
(``OpportunityAngle[]`` + carried-over signals) through the shared agent
framework and ModelRouter (mock mode by default).
"""

from __future__ import annotations

from agent_opportunity_analysis.config import (
    OPPORTUNITY_ANALYSIS_SLUG,
    OPPORTUNITY_OUTPUT_SCHEMA,
    build_opportunity_analysis_config,
)
from agent_opportunity_analysis.agent import OpportunityAnalysisAgent
from agent_opportunity_analysis.version import __version__

__all__ = [
    "__version__",
    "OPPORTUNITY_ANALYSIS_SLUG",
    "OPPORTUNITY_OUTPUT_SCHEMA",
    "OpportunityAnalysisAgent",
    "build_opportunity_analysis_config",
]
