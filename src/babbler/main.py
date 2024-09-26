"""The entrypoint that creates the CLI app."""

import typer

import babbler

typer_app = typer.Typer(
    add_completion=False,
    help='A tool for using generative APIs.',
)


def main() -> None:
    """Create and run the CLI app."""
    typer_app.command(name='complete')(babbler.chats.complete.complete_file)
    typer_app()


if __name__ == '__main__':
    main()
