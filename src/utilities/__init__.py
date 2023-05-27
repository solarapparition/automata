"""Utilities for the package."""

from datetime import datetime

def generate_timestamp_id() -> str:
    """Generate an id based on the current timestamp."""
    return datetime.utcnow().strftime("%Y-%m-%d_%H%M-%S-%f")
