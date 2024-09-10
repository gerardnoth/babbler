"""A module for utility functions."""

import pathlib


def project_root() -> pathlib.Path:
    """Get a path to the project root.

    :return: A directory path.
    """
    return pathlib.Path(__file__).parent.parent.parent
