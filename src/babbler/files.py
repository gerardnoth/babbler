"""Utilities for working with files."""

from typing import Iterable, Any

import orjson

from babbler.types import PathLike


def yield_lines(path: PathLike) -> Iterable[str]:
    """Yield lines as strings from a file.

    :param path: A path to a file.
    :return: An iterable of lines.
    """
    with open(path, encoding='utf-8') as file:
        for line in file:
            yield line


def yield_jsonl(path: PathLike) -> Iterable[Any]:
    """Yield JSON objects from a file.

    :param path: A path to a file.
    :return: An iterable of JSON objects.
    """
    with open(path, 'rb') as file:
        for line in file:
            yield orjson.loads(line)
