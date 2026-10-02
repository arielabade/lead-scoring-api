"""Request and response contracts.

The API validates against the categories the model was actually trained on.
An unknown job title is rejected at the edge with a clear message rather than
silently scored as an unseen category.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Job = Literal[
    "admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired",
    "self-employed", "services", "student", "technician", "unemployed", "unknown",
]
Marital = Literal["divorced", "married", "single", "unknown"]
Education = Literal[
    "basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
    "professional.course", "university.degree", "unknown",
]
YesNoUnknown = Literal["yes", "no", "unknown"]
Contact = Literal["cellular", "telephone"]
Month = Literal["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
DayOfWeek = Literal["mon", "tue", "wed", "thu", "fri"]
Outcome = Literal["failure", "nonexistent", "success"]


class Lead(BaseModel):
    """One lead, as known BEFORE the call.

    `duration` is deliberately absent: it is the length of the call and does not
    exist at scoring time. Accepting it would let a caller leak the outcome into
    the prediction.
    """

    age: int = Field(ge=17, le=120)
    job: Job
    marital: Marital
    education: Education
    default: YesNoUnknown
    housing: YesNoUnknown
    loan: YesNoUnknown
    contact: Contact
    month: Month
    day_of_week: DayOfWeek
    campaign: int = Field(ge=1, description="contacts during this campaign, including this one")
    pdays: int = Field(ge=0, description="days since last contact; 999 means never contacted")
    previous: int = Field(ge=0, description="contacts before this campaign")
    poutcome: Outcome


class ScoredLead(BaseModel):
    probability: float = Field(ge=0.0, le=1.0)
    call: bool
    priority: Literal["high", "medium", "low"]


class BatchRequest(BaseModel):
    leads: list[Lead] = Field(min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    scored: list[ScoredLead]
    threshold: float


class Health(BaseModel):
    status: str
    model_loaded: bool
    threshold: float | None = Field(default=None, description="break-even probability: cost / value")
