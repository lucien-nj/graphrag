# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""CLI implementation of the server subcommand."""

from pathlib import Path

import typer

from graphrag.api_server.app import create_app
from graphrag.api_server.config import (
    DEFAULT_COMMUNITY_LEVEL,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_RESPONSE_TYPE,
)


def server_cli(
    root: Path = typer.Option(
        Path.cwd(),
        "--root",
        "-r",
        help="The project root directory.",
        exists=True,
        dir_okay=True,
        file_okay=False,
        writable=True,
        resolve_path=True,
    ),
    host: str = typer.Option(
        DEFAULT_HOST,
        "--host",
        "-h",
        help="The host to bind the server to.",
    ),
    port: int = typer.Option(
        DEFAULT_PORT,
        "--port",
        "-p",
        help="The port to bind the server to.",
    ),
    community_level: int = typer.Option(
        DEFAULT_COMMUNITY_LEVEL,
        "--community-level",
        help="Leiden hierarchy level from which to load community reports.",
    ),
    response_type: str = typer.Option(
        DEFAULT_RESPONSE_TYPE,
        "--response-type",
        help="Free-form description of the desired response format.",
    ),
    data: Path | None = typer.Option(
        None,
        "--data",
        "-d",
        help="Index output directory (contains the parquet files).",
        exists=True,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
) -> None:
    """Start the GraphRAG API server."""
    import uvicorn

    app = create_app(
        root=root,
        data_dir=data,
        community_level=community_level,
        response_type=response_type,
    )
    uvicorn.run(app, host=host, port=port)
