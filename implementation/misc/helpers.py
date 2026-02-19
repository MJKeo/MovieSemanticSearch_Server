"""
Helper functions for lexical search operations.

This module contains utility functions used across the lexical search system,
including string normalization for dictionary lookups.
"""

from typing import Optional

import unicodedata
import re
import hashlib


def normalize_string(text: str) -> str:
    """
    Normalize a string for lexical dictionary lookups.
    
    Applies the following transformations in order:
    1. Unicode NFC normalization
    2. Lowercase (Unicode-aware case folding)
    3. Diacritic/accent removal (é → e, ñ → n, etc.)
    4. Punctuation handling:
       - Hyphens (-) → preserved (kept as-is for hyphenated terms like "spider-man")
       - Apostrophes (') → removed (no space)
       - Periods (.) → removed (no space)
       - All other punctuation → space
    5. Collapse multiple spaces to single space
    6. Trim leading/trailing whitespace
    
    Args:
        text: The input string to normalize. Must be a string.
        
    Returns:
        The normalized string, ready for dictionary lookup.
        Returns empty string if input is empty or whitespace-only.
        
    Raises:
        Any exceptions from underlying string operations are propagated
        to the caller (e.g., if text is not actually a string).
        
    Examples:
        >>> normalize_string("Spider-Man")
        'spider-man'
        >>> normalize_string("Ocean's Eleven")
        'oceans eleven'
        >>> normalize_string("Jean-Luc Picard")
        'jean-luc picard'
        >>> normalize_string("Amélie")
        'amelie'
        >>> normalize_string("Se7en")
        'se7en'
        >>> normalize_string("The Lord of the Rings: The Two Towers")
        'the lord of the rings the two towers'
        >>> normalize_string("L.A. Confidential")
        'la confidential'
        >>> normalize_string("A-Team")
        'a-team'
    """
    # Handle empty string edge case
    if not text:
        return ""
    
    # Step 1: Unicode NFC normalization
    # Ensures consistent representation of composed characters
    normalized = unicodedata.normalize("NFC", text)
    
    # Step 2: Lowercase (Unicode-aware case folding)
    # casefold() is more aggressive than lower() for Unicode
    normalized = normalized.casefold()
    
    # Step 3: Diacritic/accent removal
    # Decompose characters into base + combining marks (NFD), then remove combining marks
    normalized = unicodedata.normalize("NFD", normalized)
    normalized = "".join(
        char for char in normalized 
        if unicodedata.category(char) != "Mn"  # Mn = Mark, Nonspacing (combining diacritics)
    )
    
    # Step 4: Punctuation handling
    # 4a: Hyphens are PRESERVED (kept as-is)
    # This allows "spider-man" to stay as "spider-man" for exact matching
    # Hyphen expansion to ["spider-man", "spider", "man"] happens during tokenization
    
    # 4b: Apostrophes → removed (no space) (handles various Unicode apostrophes)
    normalized = re.sub(r"[''ʼ`']", "", normalized)
    
    # 4c: Periods → removed (no space)
    # This ensures "L.A." becomes "la" not "l a", matching how users search
    normalized = re.sub(r"\.", "", normalized)
    
    # 4d: All other punctuation → space (but NOT hyphens)
    # Match any character that is not alphanumeric, whitespace, or hyphen
    normalized = re.sub(r"[^\w\s\-]", " ", normalized)
    
    # Step 5: Collapse multiple spaces to single space
    normalized = re.sub(r"\s+", " ", normalized)
    
    # Step 6: Trim leading/trailing whitespace
    normalized = normalized.strip()
    
    return normalized

def create_watch_provider_offering_key(provider_id: int, watch_method_id: int) -> int:
    """
    Create a watch provider offering integer from a provider name ID and watch method ID.

    Upper 27 bits hold the provider name, lower 4 bits hold the watch method
    """
    return (provider_id << 4) | watch_method_id
