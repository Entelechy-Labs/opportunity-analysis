# opportunity-analysis

Strategy-phase agent of the **LinkGen** content-intelligence and publishing
system.

## Purpose

The specialist agent responsible for identifying content gaps and
opportunities: it consumes the research phase outputs (`ResearchRequest` plus
optional `CreatorProfile[]` / `MarketSignal[]`) and produces a
contract-conforming `OpportunityMap` — original `OpportunityAngle[]` plus the
carried-over, validated market signals.

Design highlights (see `AGENTS.md` in the parent for the master rules):

- **Deterministic by construction** — ids are derived with uuid5
  (`agent_framework.ids.deterministic_id`), time comes from the injected
  context clock, and outputs are pure functions of the input. The same
  request always yields byte-identical artifacts.
- **LLM output is untrusted** — model JSON arrives router-validated and is
  sanitized field-by-field before entering contract models; unusable entries
  (missing title/description) are dropped, duplicates deduped by normalized
  title, entries capped.
- **Focus-derived baseline** — when the model yields no usable angle, the
  agent emits a single angle derived from the request focus itself, so the
  strategy phase always receives at least one input-derived angle instead of
  an empty map (never a fabricated claim).
- **Mock-first ($0 development)** — the default provider is the offline
  deterministic `MockLLMProvider` via the `ModelRouter`. No network, no
  credentials, no cost.
- **No privileged tools** — no publishing/scheduling authority.

## Input / output contract

Input (validated against research-phase contracts):

```json
{"request": {"request_id": "...", "run_id": "...", "focus": "..."},
 "creators": [{...CreatorProfile...}],
 "signals": [{...MarketSignal...}]}
```

`creators` and `signals` are optional; `request` is required.

Output:

```json
{"opportunity_map": {...OpportunityMap...},
 "opportunities": [{...OpportunityAngle...}]}
```

## Boundaries

- Output contracts: `OpportunityMap`, `OpportunityAngle`.
- Forbidden tools: `publish`, `schedule`, `linkedin-post`.
- Depends only on `linkgen_contracts`, `linkgen_config`, `agent-framework`,
  and the `ModelRouter` interface from `providers/llm`.

## Testing

Run local tests with:

    python -m pytest

46 tests.

## Status

Phase 09 (strategy system) — implemented at 1.0.0.

## License

Proprietary — © Entelechy Labs. All rights reserved. No license is granted
to use, copy, modify, or distribute this software, except as agreed in
writing with the copyright holder.

## Publishing authority

No publishing/scheduling authority; publishing privilege belongs to the
publishing subsystem (publisher) only.
