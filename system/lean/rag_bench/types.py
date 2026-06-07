"""Schema for the rag_bench harness."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

_Frozen = ConfigDict(frozen=True, extra="forbid")


class Query(BaseModel):
    """One benchmark row: query + oracle context + gold answer + domain tag."""

    model_config = _Frozen

    query_id: str
    domain: str = Field(description="topical label, e.g. 'medical' / 'finance' / 'legal'")
    query: str
    gold_context: str
    gold_answer: str


class Subset(BaseModel):
    """A stratified, deduplicated subset built from a raw JSONL."""

    model_config = _Frozen

    queries: tuple[Query, ...]

    @property
    def domains(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for q in self.queries:
            seen.setdefault(q.domain, None)
        return tuple(seen)

    @model_validator(mode="after")
    def _non_empty(self) -> Subset:
        if not self.queries:
            raise ValueError("Subset must have at least one query")
        return self


class Generation(BaseModel):
    """One generated answer with provenance — written to the score artifact."""

    model_config = _Frozen

    query_id: str
    domain: str
    role: str  # 'd' / 's' / 'y'
    config_id: str
    answer: str
    wall_s: float = Field(ge=0)
    n_input_tokens: int = Field(ge=0)
    n_output_tokens: int = Field(ge=0)


class Generator(Protocol):
    """Anything that turns a Query into an answer text for a given (role, config).

    Real implementations: vLLM server, llama.cpp HTTP, OpenAI-compatible. The
    harness ships a `MockGenerator` so CI runs without GPU/network.
    """

    def generate(self, query: Query, role: str, config_id: str) -> Generation:  # pragma: no cover
        ...
