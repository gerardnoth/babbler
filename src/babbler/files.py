"""Utilities for working with files."""

from pathlib import Path
from typing import Iterable, Any, Self, BinaryIO

import orjson
from pydantic import BaseModel

from babbler.types import PathLike


class JSONLWriter:
    """Writes object to a file in JSONL format.

    The file parent directories are created if they do not exist.
    """

    def __init__(self, path: PathLike, append: bool = False) -> None:
        """Create a new writer.

        :param path: A path to a file to write to.
        :param append: Whether to append to an existing file.
        """
        self.path = Path(path)
        self.append = append
        self.file: BinaryIO | None = None

    def __enter__(self) -> Self:
        """Creates parent directories and opens the file for writing.

        :return: The current instance.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = 'ab' if self.append else 'wb'
        self.file = open(self.path, mode)
        return self

    def write(self, item: Any) -> None:
        """Write an object to the file.

        A newline character is appended to the end of the file.

        :param item: An object to serialize and write.
        """
        if self.file is None:
            raise ValueError('Cannot write before opening the file.')
        data: bytes
        if isinstance(item, BaseModel):
            data = item.model_dump_json().encode('utf-8')
        else:
            data = orjson.dumps(item)
        self.file.write(data)
        self.file.write(b'\n')

    def write_many(self, iterable: Iterable[Any]) -> None:
        """Writes each object in the iterable on a separate line as JSON.

        :param iterable: An iterable of objects to write.
        """
        for obj in iterable:
            self.write(obj)

    def __exit__(self, *args: Any) -> None:
        """Closes the file if it is open."""
        if self.file is not None:
            self.file.close()
            self.file = None


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
