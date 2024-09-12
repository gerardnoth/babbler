"""A module for various application resources."""

from enum import Enum
from typing import Self, Iterable

from pydantic import BaseModel

import babbler
from babbler.types import PathLike


class JsonModel(BaseModel):
    """An extension of a Pydantic BaseModel with additional JSON methods."""

    @classmethod
    def from_json(cls, path: PathLike) -> Self:
        """Create an instance from a JSON file.

        :param path: Path to a JSON file.
        :return: A new instance.
        """
        with open(path, encoding='utf-8') as file:
            return cls.model_validate_json(file.read())

    @classmethod
    def from_jsonl(cls, path: PathLike) -> list[Self]:
        """Reads objects from a JSONL file and returns them as a list.

        :param path: Path to a file.
        :return: A list of instances.
        """
        return [x for x in cls.yield_from_jsonl(path)]

    @classmethod
    def yield_from_jsonl(cls, path: PathLike) -> Iterable[Self]:
        """Yields deserialized instances from a JSONL file.

        :param path: Path to a file.
        :return: An iterable of instances.
        """
        for line in babbler.files.yield_lines(path):
            yield cls.model_validate_json(line)


class Provider(str, Enum):
    """Indicates supported model providers."""

    google_ai = 'google_ai'
    openai = 'openai'
