"""
Shared SQL LIKE helpers.

Utilities in this module centralize escaping rules for SQL LIKE patterns so
all callers use consistent behavior.
"""

import re

# Pre-compiled regex for escaping LIKE pattern metacharacters.
_LIKE_ESCAPE_RE = re.compile(r"([\\%_])")


def escape_like(value: str) -> str:
    r"""
    Escape SQL LIKE metacharacters so *value* is treated as a literal substring.

    Uses ``\`` as the SQL LIKE escape character.
    """
    return _LIKE_ESCAPE_RE.sub(r"\\\1", value)
