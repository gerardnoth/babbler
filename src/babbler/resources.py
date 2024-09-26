"""A module for various application resources."""

from enum import Enum
from pathlib import Path
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
    def yield_from_jsonl(cls, path: PathLike, limit: int | None = None) -> Iterable[Self]:
        """Yields deserialized instances from a JSONL file.

        :param path: Path to a file.
        :param limit: The maximum number of instances to yield.
        :return: An iterable of instances.
        """
        for line in babbler.files.yield_lines(path, limit=limit):
            yield cls.model_validate_json(line)

    def write_json(self, path: PathLike) -> None:
        """Serialize as JSON and write the result to a file.

        :param path: Path to save to.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as file:
            file.write(self.model_dump_json())


class Provider(str, Enum):
    """Indicates supported model providers."""

    google_ai = 'google_ai'
    openai = 'openai'
    vertexai = 'vertexai'
