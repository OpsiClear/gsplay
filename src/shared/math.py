"""Mathematical utilities for the universal 4D viewer."""

import re
from pathlib import Path


def natural_sort_key(path: str | Path) -> list:
    """
    Generate a natural sort key for filenames with numeric sequences.

    Handles cases like:
    - image_1, image_2, ..., image_10 (not image_1, image_10, image_2)
    - frame_001, frame_002, ..., frame_100
    - file1.txt, file2.txt, ..., file10.txt

    Args:
        path: File path or filename to generate sort key for

    Returns:
        List of mixed str/int for natural sorting

    Example:
        >>> sorted(['image_1.ply', 'image_10.ply', 'image_2.ply'], key=natural_sort_key)
        ['image_1.ply', 'image_2.ply', 'image_10.ply']
    """
    path_str = str(path)
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', path_str)]
