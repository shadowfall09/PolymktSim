"""Data models: EvidenceItem, Forecast, QuestionExample."""
from __future__ import annotations
from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    doc_id: str
    source: str = ""
    timestamp: Optional[datetime] = None
    title: Optional[str] = None
    url: Optional[str] = None
    content: str = ""
    retrieval_score: Optional[float] = None


class Forecast(BaseModel):
    p_yes: float = Field(ge=0, le=1)
    label: str = Field(pattern="^(YES|NO)$")
    rationale: str = ""
    evidence_used: list[str] = Field(default_factory=list)

    @classmethod
    def from_p_yes(cls, p_yes: float, rationale: str = "", evidence_used: list[str] | None = None):
        label = "YES" if p_yes >= 0.5 else "NO"
        return cls(p_yes=p_yes, label=label, rationale=rationale, evidence_used=evidence_used or [])


class QuestionExample(BaseModel):
    qid: str
    question: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    outcome: Optional[bool] = None  # True=Yes, False=No, None=unresolved
    resolution_date: Optional[date] = None  # market end date for temporal scoring
