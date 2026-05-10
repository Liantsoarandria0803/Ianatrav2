import logging

from sqlalchemy import text

from app.core.database import AsyncSessionLocal, Base, engine

logger = logging.getLogger(__name__)


async def _ensure_enum_type(conn, type_name: str, values: list[str]) -> None:
    """Crée le type ENUM Postgres s'il n'existe pas.

    Objectif: éviter les mismatches de schéma entre environnements.
    """
    escaped_values = ",".join([f"'{v}'" for v in values])
    await conn.execute(
        text(
            f"""
            DO $$
            BEGIN
              IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = '{type_name}') THEN
                CREATE TYPE {type_name} AS ENUM ({escaped_values});
              END IF;
            END$$;
            """
        )
    )


async def _get_column_pg_type(conn, table: str, column: str) -> str | None:
    res = await conn.execute(
        text(
            """
            SELECT t.typname
            FROM pg_attribute a
            JOIN pg_class c ON a.attrelid = c.oid
            JOIN pg_type t ON a.atttypid = t.oid
            JOIN pg_namespace n ON c.relnamespace = n.oid
            WHERE n.nspname = 'public'
              AND c.relname = :table
              AND a.attname = :column
              AND a.attnum > 0
              AND NOT a.attisdropped
            """
        ),
        {"table": table, "column": column},
    )
    return res.scalar_one_or_none()


async def _align_enum_column_type(
    conn,
    *,
    table: str,
    column: str,
    expected_type: str,
    set_default_sql: str | None = None,
) -> None:
    current_type = await _get_column_pg_type(conn, table, column)
    if not current_type:
        return
    if current_type == expected_type:
        return

    logger.warning(
        "Align enum column type: %s.%s (%s -> %s)",
        table,
        column,
        current_type,
        expected_type,
    )

    # Defaults can be typed as the old enum; drop it before changing types.
    await conn.execute(text(f'ALTER TABLE "{table}" ALTER COLUMN "{column}" DROP DEFAULT'))
    await conn.execute(
        text(
            f'ALTER TABLE "{table}" ALTER COLUMN "{column}" '
            f'TYPE {expected_type} '
            f'USING "{column}"::text::{expected_type}'
        )
    )
    if set_default_sql:
        await conn.execute(text(f'ALTER TABLE "{table}" ALTER COLUMN "{column}" SET DEFAULT {set_default_sql}'))


async def _align_schema(conn) -> None:
    """Aligne les colonnes ENUM vers les noms attendus par l'application."""
    await _align_enum_column_type(conn, table="users", column="vertical", expected_type="vertical_enum")
    await _align_enum_column_type(
        conn,
        table="knowledge_chunks",
        column="vertical",
        expected_type="vertical_enum",
        set_default_sql="'concours'::vertical_enum",
    )
    await _align_enum_column_type(conn, table="sessions", column="mode", expected_type="session_mode_enum")
    await _align_enum_column_type(conn, table="messages", column="role", expected_type="message_role_enum")


async def init_database() -> None:
    """Initialise pgvector, les tables SQLAlchemy et l'index vectoriel."""
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        # Important: create_all() ne crée pas forcément les types ENUM si les tables existent déjà.
        await _ensure_enum_type(conn, "vertical_enum", ["concours", "bac", "prepa"])
        await _ensure_enum_type(conn, "session_mode_enum", ["cours", "exercice", "quiz"])
        await _ensure_enum_type(conn, "message_role_enum", ["user", "assistant"])

        await conn.run_sync(Base.metadata.create_all)

        # Auto-heal: si une version précédente a créé des enums avec d'autres noms,
        # on convertit les colonnes pour éviter les 500 DatatypeMismatchError.
        await _align_schema(conn)
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_chunks_embedding_hnsw
            ON knowledge_chunks
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """))
    logger.info("Base de donnees initialisee avec pgvector + index HNSW")


async def ingest_initial_corpus() -> int:
    """Ingère le corpus RAG initial si les chunks ne sont pas déjà présents."""
    from app.rag.rag_service import RAGService

    async with AsyncSessionLocal() as db:
        service = RAGService(db)
        count = await service.ingest()
        await db.commit()
        logger.info("Corpus RAG initialise: %s nouveaux chunks", count)
        return count
