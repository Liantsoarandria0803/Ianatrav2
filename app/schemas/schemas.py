from datetime import date, datetime, time, timezone
from typing import Any, Optional
from uuid import UUID
from pydantic import BaseModel, EmailStr, Field, field_validator
from app.models.models import VerticalEnum, SessionModeEnum, MessageRoleEnum


# ── Auth ──────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = None
    vertical: VerticalEnum
    exam_date: Optional[datetime] = None

    @field_validator("email", mode="before")
    @classmethod
    def normalize_email(cls, v: Any) -> Any:
        return v.strip().lower() if isinstance(v, str) else v

    @field_validator("name", mode="before")
    @classmethod
    def normalize_name(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = v.strip()
            return v or None
        return v

    @field_validator("exam_date", mode="before")
    @classmethod
    def parse_exam_date(cls, v: Any) -> Any:
        if v in ("", None):
            return None
        if isinstance(v, datetime):
            return v
        if isinstance(v, date):
            return datetime.combine(v, time.min, tzinfo=timezone.utc)
        if isinstance(v, str) and len(v) == 10:
            return datetime.combine(date.fromisoformat(v), time.min, tzinfo=timezone.utc)
        return v

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Le mot de passe doit contenir au moins 8 caractères")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str

    @field_validator("email", mode="before")
    @classmethod
    def normalize_email(cls, v: Any) -> Any:
        return v.strip().lower() if isinstance(v, str) else v


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: UUID
    email: str
    name: Optional[str]
    vertical: VerticalEnum
    exam_date: Optional[datetime]
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Diagnostic ────────────────────────────────────────────────────────────────

class DiagnosticQuestion(BaseModel):
    id: int
    topic: str
    question: str
    type: str = "open"  # open | qcm


class DiagnosticStartResponse(BaseModel):
    diagnostic_id: str
    questions: list[DiagnosticQuestion]


class DiagnosticAnswer(BaseModel):
    question_id: int
    answer: str


class DiagnosticSubmitRequest(BaseModel):
    diagnostic_id: str
    answers: list[DiagnosticAnswer]


class TopicScore(BaseModel):
    topic: str
    score: float
    status: str  # maîtrisé | lacune | moyen


class DiagnosticSubmitResponse(BaseModel):
    scores: list[TopicScore]
    summary: str
    strong_topics: list[str]
    weak_topics: list[str]


# ── Session ───────────────────────────────────────────────────────────────────

class SessionStartRequest(BaseModel):
    mode: SessionModeEnum
    topic: Optional[str] = None


class SessionStartResponse(BaseModel):
    # ORM attribute is `Session.id`, but the API contract is `session_id`.
    session_id: UUID = Field(validation_alias="id")
    mode: SessionModeEnum
    topic: Optional[str]
    started_at: datetime

    model_config = {"from_attributes": True}


class MessageRequest(BaseModel):
    content: str


class MessageResponse(BaseModel):
    id: UUID
    role: MessageRoleEnum
    content: str
    tokens_used: int
    created_at: datetime

    model_config = {"from_attributes": True}


class SessionHistoryResponse(BaseModel):
    session_id: UUID
    messages: list[MessageResponse]
    total_tokens: int


# ── Competences ───────────────────────────────────────────────────────────────

class CompetenceResponse(BaseModel):
    topic: str
    score: float
    status: str  # maîtrisé | lacune | moyen | non évalué
    review_interval_days: int
    last_updated: datetime

    model_config = {"from_attributes": True}


class CompetenceProfileResponse(BaseModel):
    competences: list[CompetenceResponse]
    days_until_exam: Optional[int]
