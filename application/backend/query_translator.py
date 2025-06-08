"""
Query Translator Module

This module translates natural language queries about NYC 311 data
into structured query definitions for the database engine.
"""

# Standard library imports
import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

# Third-party library imports
from google.genai import types

# Local imports
from utils import (
    BASE_DIR, DATA_SCHEMA_FILE, FILTER_VALUES_FILE,
    get_gemini_client, call_gemini_async, gemini_safe
)

# === CONSTANTS ===
INSTRUCTION_FILE = os.path.join(BASE_DIR, "gemini_instructions/query_translation_instruction.md")

# === GLOBAL CONFIGURATION ===
logger = logging.getLogger(__name__)

# Module state
_client = None
_system_instruction = None
_schema = None


# === HELPER FUNCTIONS ===
def _initialize() -> None:
    """
    Initialize the query translator with API client and instructions.
    
    This function loads the system instruction, data schema, and filter values.
    It runs only once during the module lifetime and caches the results.
    """
    global _client, _system_instruction, _schema
       
    # Only initialize once
    if _system_instruction is not None:
        return
        
    # Load API client
    _client = get_gemini_client()
    
    # Load system instruction
    with open(INSTRUCTION_FILE, "r") as f:
        _system_instruction = f.read()
    
    # Load schema
    with open(DATA_SCHEMA_FILE, "r") as f:
        _schema = json.load(f)
        
    # Load filter values
    with open(FILTER_VALUES_FILE, "r") as f:
        all_filters = json.load(f)
        
    # Replace placeholders in the instruction
    _system_instruction = _system_instruction.replace(
        "{data_schema}", json.dumps(_schema)
    )
    _system_instruction = _system_instruction.replace(
        "{all_filters}", json.dumps(all_filters)
    )
    
    logger.info("Query translator initialized")


# === MAIN FUNCTIONALITY ===
@gemini_safe
async def translate_query(raw_query: str, context: Optional[Dict[str, Any]] = None) -> Tuple[str, bool]:
    """
    Translate natural language query into a structured aggregation definition.
    
    Args:
        raw_query: The natural language query from the user
        context: Optional conversation context including history
        
    Returns:
        Tuple containing:
        - The structured query definition or direct response text
        - Boolean flag indicating if this is a direct response (True) or query definition (False)
    """
    # Ensure system is initialized
    _initialize()
    
    # Prepare full prompt with context if available
    full_prompt = raw_query
    if context and context.get("conversationHistory"):
        context_prompt = json.dumps(context, indent=2)
        full_prompt = f"USER_QUERY:\n{raw_query}\n\n\n\nCONTEXT:\n{context_prompt}"
            
    # Log query info
    logger.info(f"Translating query: {full_prompt}")
    
    # Call Gemini API
    response = await call_gemini_async(
        "gemini-2.5-flash-preview-05-20",
        full_prompt,
        config=types.GenerateContentConfig(
            system_instruction=_system_instruction,
            temperature=0,
        )
    )
    
    # Extract response text
    response_text = response.candidates[0].content.parts[0].text.strip()
    
    # Handle direct responses that don't need query translation
    if response_text.startswith("DIRECT_RESPONSE:"):
        direct_response = response_text[len("DIRECT_RESPONSE:"):].strip()
        logger.info("Query translator provided direct response")
        return direct_response, True
        
    # Log success and return the structured query definition
    logger.debug(f"Query translation successful, response length: {len(response_text)}")
    return response_text, False