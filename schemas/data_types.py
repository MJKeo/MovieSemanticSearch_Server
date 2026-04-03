"""
Shared data types used across multiple modules in the codebase.
"""


class MultiLineList(list):
    """Marker for lists whose items should be formatted on separate lines.

    When passed to build_user_prompt, items are rendered as:
        key:
        - item 1
        - item 2

    Regular lists are comma-separated on a single line. Use MultiLineList
    for long-text items like plot_summaries and plot_synopses where each
    entry is a multi-sentence block.
    """
    pass
