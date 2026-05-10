import uuid
from datetime import datetime, timezone
from enum import Enum as PyEnum
from typing import Optional
from sqlalchemy import (
    String, Text, Float, Integer, DateTime, ForeignKey, JSON, Enum, Index, func
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
from app.core.database import Base


def utcnow():
    return datetime.now(timezone.utc)


# ── Enums ─────────────────────────────────────────────────────────────────────

class VerticalEnum(str, PyEnum):
    concours = "concours"
    bac = "bac"
    prepa = "prepa"


class SessionModeEnum(str, PyEnum):
    cours = "cours"
    exercice = "exercice"
    quiz = "quiz"


class MessageRoleEnum(str, PyEnum):
    user = "user"
    assistant = "assistant"


# ── Models ────────────────────────────────────────────────────────────────────

class User(Base):
    """
    Entité utilisateur — multi-tenant par 'vertical'
    Index sur email pour les lookups d'auth (unique)
    """
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[Optional[str]] = mapped_column(String(100))
    # NOTE: Keep the PostgreSQL enum type name stable across environments.
    # Render DB already has vertical_enum, so we must use the same name here.
    vertical: Mapped[VerticalEnum] = mapped_column(Enum(VerticalEnum, name="vertical_enum"), nullable=False)
    exam_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    # Relations
    competence_profiles: Mapped[list["CompetenceProfile"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    sessions: Mapped[list["Session"]] = relationship(back_populates="user", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_users_vertical", "vertical"),
    )


class CompetenceProfile(Base):
    """
    Profil de compétences par topic avec algorithme SM-2 (spaced repetition)
    Index composé (user_id, topic) pour les requêtes fréquentes
    """
    __tablename__ = "competence_profiles"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    topic: Mapped[str] = mapped_column(String(100), nullable=False)
    score: Mapped[float] = mapped_column(Float, default=50.0)  # 0-100
    last_updated: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    review_interval_days: Mapped[int] = mapped_column(Integer, default=1)  # SM-2

    # Relation
    user: Mapped["User"] = relationship(back_populates="competence_profiles")

    __table_args__ = (
        Index("ix_competence_user_topic", "user_id", "topic", unique=True),
        Index("ix_competence_score", "score"),
    )


class Session(Base):
    """
    Session pédagogique — contient l'historique des messages et les métadonnées
    Index sur user_id + started_at pour les requêtes d'historique
    """
    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    mode: Mapped[SessionModeEnum] = mapped_column(Enum(SessionModeEnum, name="session_mode_enum"), nullable=False)
    topic: Mapped[Optional[str]] = mapped_column(String(100))
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    summary: Mapped[Optional[str]] = mapped_column(Text)  # Résumé compressé LLM

    # Relation
    user: Mapped["User"] = relationship(back_populates="sessions")
    messages: Mapped[list["Message"]] = relationship(back_populates="session", cascade="all, delete-orphan", order_by="Message.created_at")

    __table_args__ = (
        Index("ix_sessions_user_date", "user_id", "started_at"),
        Index("ix_sessions_topic", "topic"),
    )


class Message(Base):
    """
    Message individuel dans une session
    Index sur session_id + created_at pour l'historique ordonné
    """
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    role: Mapped[MessageRoleEnum] = mapped_column(Enum(MessageRoleEnum, name="message_role_enum"), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    # Relation
    session: Mapped["Session"] = relationship(back_populates="messages")

    __table_args__ = (
        Index("ix_messages_session_date", "session_id", "created_at"),
    )


class KnowledgeChunk(Base):
    """
    Chunk du corpus RAG vectorisé avec pgvector
    Embedding all-MiniLM-L6-v2 = 384 dimensions
    Index HNSW pour la recherche ANN (Approximate Nearest Neighbor)
    """
    __tablename__ = "knowledge_chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chunk_id: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)  # C001, C002...
    topic: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    vertical: Mapped[VerticalEnum] = mapped_column(Enum(VerticalEnum, name="vertical_enum"), default=VerticalEnum.concours)
    embedding: Mapped[list[float]] = mapped_column(Vector(384))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        Index("ix_chunks_topic_vertical", "topic", "vertical"),
        # L'index HNSW est créé via migration Alembic (SQL brut)
    )


class DiagnosticSession(Base):
    """Session de diagnostic persistée (résiliente aux redémarrages).

    Stocke les questions envoyées à l'utilisateur afin de pouvoir valider la
    soumission même si le serveur redémarre (cas fréquent sur Render free).
    """

    __tablename__ = "diagnostic_sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    questions: Mapped[list[dict]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    consumed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("ix_diag_sessions_user_created", "user_id", "created_at"),
        Index("ix_diag_sessions_expires", "expires_at"),
    )
