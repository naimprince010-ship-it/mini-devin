"""Add session workspace and conversation history columns

Revision ID: b7d2c6f31a9d
Revises: a4fa26d3994c
Create Date: 2026-05-26 09:58:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "b7d2c6f31a9d"
down_revision: Union[str, Sequence[str], None] = "a4fa26d3994c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    bind = op.get_bind()
    insp = sa.inspect(bind)
    tables = set(insp.get_table_names())
    if "sessions" not in tables:
        return

    col_names = {c["name"] for c in insp.get_columns("sessions")}
    if "workspace_id" not in col_names:
        op.add_column("sessions", sa.Column("workspace_id", sa.String(length=64), nullable=True))
    if "conversation_json" not in col_names:
        op.add_column("sessions", sa.Column("conversation_json", sa.Text(), nullable=True))

    indexes = {idx["name"] for idx in insp.get_indexes("sessions")}
    if "ix_sessions_workspace_id" not in indexes:
        op.create_index("ix_sessions_workspace_id", "sessions", ["workspace_id"], unique=True)


def downgrade() -> None:
    """Downgrade schema."""
    bind = op.get_bind()
    insp = sa.inspect(bind)
    tables = set(insp.get_table_names())
    if "sessions" not in tables:
        return

    indexes = {idx["name"] for idx in insp.get_indexes("sessions")}
    if "ix_sessions_workspace_id" in indexes:
        op.drop_index("ix_sessions_workspace_id", table_name="sessions")

    col_names = {c["name"] for c in insp.get_columns("sessions")}
    if "conversation_json" in col_names:
        op.drop_column("sessions", "conversation_json")
    if "workspace_id" in col_names:
        op.drop_column("sessions", "workspace_id")
