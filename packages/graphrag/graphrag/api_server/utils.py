# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Utility functions for the GraphRAG API server."""

import json
from typing import Any

import pandas as pd


def convert_response_to_string(
    response: str | dict[str, Any] | list[dict[str, Any]],
) -> str:
    """Convert a response to a string."""
    if isinstance(response, (dict, list)):
        return json.dumps(response)
    elif isinstance(response, str):
        return response
    else:
        return str(response)


def recursively_convert(obj: Any) -> Any:
    """Recursively convert DataFrames to dicts."""
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, list):
        return [recursively_convert(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: recursively_convert(value) for key, value in obj.items()}
    return obj


def process_context_data(
    context_data: str | list[pd.DataFrame] | dict[str, pd.DataFrame] | pd.DataFrame,
) -> Any:
    """Normalize context data into a JSON-serializable form."""
    if isinstance(context_data, str):
        return context_data
    if isinstance(context_data, pd.DataFrame):
        return context_data.to_dict(orient="records")
    if isinstance(context_data, (list, dict)):
        return recursively_convert(context_data)
    return None
