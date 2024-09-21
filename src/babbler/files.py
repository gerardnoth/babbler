"""Utilities for working with files."""

import base64
import hashlib
from pathlib import Path
from typing import Iterable, Any, Self, BinaryIO, Literal

import orjson
from pydantic import BaseModel

from babbler.types import PathLike


def md5sum(
    path: PathLike,
    chunk_size: int = 8192,
    mode: Literal['hex', 'base64'] = 'hex',
) -> str:
    """Calculates the MD5 checksum of a file.

    If the chunk size is greater than 0, the file is read in chunks.

    :param path: A path to a file.
    :param chunk_size: The size of each chunk.
    :param mode: How the result should be returned.
    :return: The checksum.
    """
    md5 = hashlib.md5()
    with open(path, 'rb') as file:
        if chunk_size > 0:
            for chunk in iter(lambda: file.read(chunk_size), b''):
                md5.update(chunk)
        else:
            md5.update(file.read())
    if mode == 'hex':
        return md5.hexdigest()
    elif mode == 'base64':
        return base64.b64encode(md5.digest()).decode('utf-8')
    else:
        raise ValueError(f'Unsupported mode: {mode}')


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
        if self.append:
            self.file = open(self.path, 'ab')
        else:
            self.file = open(self.path, 'wb')
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
