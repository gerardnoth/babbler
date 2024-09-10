"""Utilities for working with files."""

import json
from typing import Iterable, Any

from babbler.types import PathLike


def yield_lines(path: PathLike) -> Iterable[str]:
    """Yield line bytes from a file.

    :param path: A path to a file.
    :return: An iterable of lines.
    """
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            yield line


def yield_jsonl(path: PathLike) -> Iterable[Any]:
    """Yield JSON objects from a file.

    :param path: A path to a file.
    :return: An iterable of JSON objects.
    """
    with open(path, 'rb') as file:
        for line in file:
            yield json.loads(line)
