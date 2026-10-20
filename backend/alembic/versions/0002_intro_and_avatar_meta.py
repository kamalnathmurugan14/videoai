"""intro settings and avatar metadata

Revision ID: 0002
Revises: 0001
"""
from alembic import op
import sqlalchemy as sa

revision = '0002'
down_revision = '0001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table('projects', schema=None) as batch_op:
        batch_op.add_column(sa.Column('intro_enabled', sa.Boolean(), nullable=False, server_default=sa.true()))
        batch_op.add_column(sa.Column('intro_title', sa.String(length=120), nullable=True))
    with op.batch_alter_table('media_assets', schema=None) as batch_op:
        batch_op.add_column(sa.Column('meta', sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('media_assets', schema=None) as batch_op:
        batch_op.drop_column('meta')
    with op.batch_alter_table('projects', schema=None) as batch_op:
        batch_op.drop_column('intro_title')
        batch_op.drop_column('intro_enabled')
