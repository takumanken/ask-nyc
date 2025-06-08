"""Utility functions used across multiple modules in the application."""

import json
import logging
import os
import asyncio
import functools
from typing import Dict, List, Tuple, Any, Union

import pandas as pd
import polars as pl
from fastapi import HTTPException
from google import genai

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Common file paths used across modules
DATA_SCHEMA_FILE = os.path.join(BASE_DIR, "data/data_schema.json")
FILTER_VALUES_FILE = os.path.join(BASE_DIR, "gemini_instructions/references/all_filters.json")

# Load dimension lists from schema file
def _load_dimensions_from_schema() -> Tuple[List[str], List[str]]:
    """
    Load time and geo dimensions from the data schema file.
    
    Returns:
        Tuple containing (time_dimensions, geo_dimensions)
    """
    try:
        with open(DATA_SCHEMA_FILE, "r") as f:
            schema = json.load(f)
            
        time_dimensions = [dim["physical_name"] for dim in schema["dimensions"]["time_dimension"]]
        geo_dimensions = [dim["physical_name"] for dim in schema["dimensions"]["geo_dimension"]]
        
        return time_dimensions, geo_dimensions
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error loading dimensions from schema: {e}")
        # Fallback to empty lists if schema can't be loaded
        return [], []

# Constants loaded from schema
TIME_DIMENSIONS, GEO_DIMENSIONS = _load_dimensions_from_schema()


# === DIMENSION CLASSIFICATION ===
def classify_dimensions(dimensions: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Classify dimensions into time, geo, and categorical types.
    
    Args:
        dimensions: List of dimension names
        
    Returns:
        Tuple of (time_dimensions, geo_dimensions, categorical_dimensions)
    """
    time_dims = [d for d in dimensions if d in TIME_DIMENSIONS]
    geo_dims = [d for d in dimensions if d in GEO_DIMENSIONS]
    # A dimension is categorical if it's not a time dimension
    # (geo dimensions can be both geo and categorical)
    cat_dims = [d for d in dimensions if d not in TIME_DIMENSIONS]
    
    return time_dims, geo_dims, cat_dims


# === DATAFRAME CONVERSION ===
def convert_to_dataframe(dataset: List[Dict]) -> Union[pl.DataFrame, pd.DataFrame]:
    """Converts dataset to a DataFrame, using Polars if possible."""
    try:
        return pl.DataFrame(dataset)
    except Exception as e:
        logger.warning(f"Failed to use Polars: {str(e)}. Using pandas.")
        return pd.DataFrame(dataset)


# === JSON HANDLING ===
def extract_json(text: str, return_status: bool = False) -> Union[Dict[str, Any], Tuple[Dict[str, Any], bool]]:
    """
    Extracts and parses JSON from text content, handling markdown code blocks.
    
    Args:
        text: Text that may contain JSON, possibly within markdown code blocks
        return_status: Whether to return a tuple with success flag (for backward compatibility)
    
    Returns:
        If return_status=False: Parsed JSON dict or empty dict on failure
        If return_status=True: Tuple of (parsed_json, is_valid_json)
    """
    is_json = text.strip().startswith("{") or "```" in text
    
    if not is_json:
        return ({}, False) if return_status else {}
    
    # Clean up JSON formatting if it's in a code block
    clean_text = text.strip()
    if "```" in clean_text:
        # Handle different markdown formats
        for delimiter in ["```json", "```"]:
            if delimiter in clean_text:
                clean_text = clean_text.split(delimiter, 1)[1]
                break
                
        if "```" in clean_text:
            clean_text = clean_text.split("```", 1)[0]
    
    clean_text = clean_text.strip()
    
    try:
        parsed_json = json.loads(clean_text)
        logger.debug(f"Successfully parsed JSON with keys: {list(parsed_json.keys())}")
        return (parsed_json, True) if return_status else parsed_json
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON: {e}")
        return ({}, False) if return_status else {}


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles objects with __dict__ attribute."""
    def default(self, obj):
        # Handle objects with __dict__ attribute (like TopNDefinition)
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        # Handle any other special cases
        return super().default(obj)


# === GEMINI API CLIENT ===
_gemini_client = None

def get_gemini_client():
    """
    Initialize and return the Gemini client using a singleton pattern.
    
    Returns:
        genai.Client: Initialized Gemini API client
    """
    global _gemini_client
    if _gemini_client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client

async def call_gemini_async(model_name: str, prompt: Union[str, List], **kwargs):
    """Simple async wrapper for Gemini API calls"""
    client = get_gemini_client()
    return await asyncio.to_thread(
        client.models.generate_content,
        model=model_name,
        contents=[prompt] if isinstance(prompt, str) else prompt,
        **kwargs
    )


# === ERROR HANDLING ===
def gemini_safe(fn):
    """Decorator for handling all Gemini API errors consistently"""
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        try:
            return await fn(*args, **kwargs)

        except genai.errors.APIError as err:
            logger = logging.getLogger('utils')
            logger.error(f"Gemini API error: {err}")

            status_code = getattr(err, "code", 500)
            if status_code == 429:
                message = "Rate limit exceeded - please try again later"
            elif 500 <= status_code < 600:
                message = "AI service temporarily unavailable - please retry"
            else:
                message = getattr(err, "message", "API error")

            raise HTTPException(
                status_code=status_code,
                detail={"error": message, "error_type": err.__class__.__name__}
            )
        
        except Exception as e:
            logger = logging.getLogger('utils')
            logger.exception(f"Unexpected error in Gemini API call: {e}")
            raise HTTPException(
                status_code=500,
                detail={"error": "An unexpected error occurred", "error_type": "ServerError"}
            )

    return wrapper


# === LOGGING UTILITIES ===

def get_logger(name=None):
    """Get a logger with the given name"""
    return logging.getLogger(name or __name__)

# Configure basic logging setup
def configure_logging():
    """Configure basic logging for the application"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


# Setup logging
logger = logging.getLogger(__name__)