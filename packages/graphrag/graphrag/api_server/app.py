# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""FastAPI application for the GraphRAG API server."""

import traceback
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from graphrag_storage import create_storage
from graphrag_storage.tables.table_provider_factory import create_table_provider

import graphrag.api as api
from graphrag.api_server.config import (
    DEFAULT_COMMUNITY_LEVEL,
    DEFAULT_RESPONSE_TYPE,
)
from graphrag.api_server.utils import process_context_data
from graphrag.config.load_config import load_config
from graphrag.data_model.data_reader import DataReader


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load index data on startup."""
    config = app.state._graphrag_config
    data_dir = app.state._graphrag_data_dir

    cli_overrides: dict[str, Any] = {}
    if data_dir:
        cli_overrides["output_storage"] = {"base_dir": str(data_dir)}

    loaded_config = load_config(
        root_dir=config["root_dir"],
        cli_overrides=cli_overrides,
    )
    app.state.config = loaded_config

    storage_obj = create_storage(loaded_config.output_storage)
    table_provider = create_table_provider(loaded_config.table_provider, storage=storage_obj)
    reader = DataReader(table_provider)

    app.state.entities = await reader.entities()
    app.state.communities = await reader.communities()
    app.state.community_reports = await reader.community_reports()
    app.state.text_units = await reader.text_units()
    app.state.relationships = await reader.relationships()

    covariates_exists = await table_provider.has("covariates")
    app.state.covariates = await reader.covariates() if covariates_exists else None

    yield


def create_app(
    root: Path,
    data_dir: Path | None = None,
    community_level: int = DEFAULT_COMMUNITY_LEVEL,
    response_type: str = DEFAULT_RESPONSE_TYPE,
) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(lifespan=lifespan)

    app.state._graphrag_config = {"root_dir": root}
    app.state._graphrag_data_dir = data_dir
    app.state.community_level = community_level
    app.state.response_type = response_type

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "https://noworneverev.github.io"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/search/global")
    async def global_search(
        query: str = Query(..., description="Global Search"),
    ) -> JSONResponse:
        try:
            response, context = await api.global_search(
                config=app.state.config,
                entities=app.state.entities,
                communities=app.state.communities,
                community_reports=app.state.community_reports,
                community_level=app.state.community_level,
                dynamic_community_selection=False,
                response_type=app.state.response_type,
                query=query,
            )
            return JSONResponse(
                content={
                    "response": response,
                    "context_data": process_context_data(context),
                }
            )
        except Exception:
            raise HTTPException(status_code=500, detail=traceback.format_exc())

    @app.get("/search/local")
    async def local_search(
        query: str = Query(..., description="Local Search"),
    ) -> JSONResponse:
        try:
            response, context = await api.local_search(
                config=app.state.config,
                entities=app.state.entities,
                communities=app.state.communities,
                community_reports=app.state.community_reports,
                text_units=app.state.text_units,
                relationships=app.state.relationships,
                covariates=app.state.covariates,
                community_level=app.state.community_level,
                response_type=app.state.response_type,
                query=query,
            )
            return JSONResponse(
                content={
                    "response": response,
                    "context_data": process_context_data(context),
                }
            )
        except Exception:
            raise HTTPException(status_code=500, detail=traceback.format_exc())

    @app.get("/search/drift")
    async def drift_search(
        query: str = Query(..., description="DRIFT Search"),
    ) -> JSONResponse:
        try:
            response, context = await api.drift_search(
                config=app.state.config,
                entities=app.state.entities,
                communities=app.state.communities,
                community_reports=app.state.community_reports,
                text_units=app.state.text_units,
                relationships=app.state.relationships,
                community_level=app.state.community_level,
                response_type=app.state.response_type,
                query=query,
            )
            return JSONResponse(
                content={
                    "response": response,
                    "context_data": process_context_data(context),
                }
            )
        except Exception:
            raise HTTPException(status_code=500, detail=traceback.format_exc())

    @app.get("/search/basic")
    async def basic_search(
        query: str = Query(..., description="Basic Search"),
    ) -> JSONResponse:
        try:
            response, context = await api.basic_search(
                config=app.state.config,
                text_units=app.state.text_units,
                response_type=app.state.response_type,
                query=query,
            )
            return JSONResponse(
                content={
                    "response": response,
                    "context_data": process_context_data(context),
                }
            )
        except Exception:
            raise HTTPException(status_code=500, detail=traceback.format_exc())

    @app.get("/status")
    async def status() -> JSONResponse:
        return JSONResponse(content={"status": "Server is up and running"})

    return app
