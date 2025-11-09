"""
Database mixins for common patterns.

Provides reusable mixins for timestamps, soft deletes, and other common patterns.
"""

from datetime import datetime
from sqlalchemy import Column, DateTime
from sqlalchemy.sql import func


class TimestampMixin:
    """Mixin for created_at and updated_at timestamps."""

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="Timestamp when the record was created",
    )

    updated_at = Column(
        DateTime(timezone=True),
        onupdate=func.now(),
        nullable=True,
        comment="Timestamp when the record was last updated",
    )


class SoftDeleteMixin:
    """Mixin for soft delete functionality."""

    deleted_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when the record was soft deleted (NULL = not deleted)",
    )

    @property
    def is_deleted(self) -> bool:
        """Check if the record is soft deleted."""
        return self.deleted_at is not None

    def soft_delete(self) -> None:
        """Mark the record as deleted."""
        self.deleted_at = datetime.utcnow()

    def restore(self) -> None:
        """Restore a soft deleted record."""
        self.deleted_at = None