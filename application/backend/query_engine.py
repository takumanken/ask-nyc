"""
Query Engine Module

Handles SQL query generation, execution, and result processing for NYC 311 data.
"""

# Standard library imports
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports
import duckdb
import pandas as pd
from google.genai import types

# Local imports
from models import AggregationDefinition
from utils import (
    BASE_DIR, DATA_SCHEMA_FILE, FILTER_VALUES_FILE, 
    TIME_DIMENSIONS, extract_json, classify_dimensions,
    call_gemini_async, gemini_safe
)

# === CONSTANTS ===
SYSTEM_INSTRUCTION_FILE = os.path.join(BASE_DIR, "gemini_instructions/data_aggregation_instruction.md")
DUCKDB_FILE = os.path.join(BASE_DIR, "data/nyc_open_data_explorer.duckdb")
_system_instruction = None

# === GLOBAL CONFIGURATION ===
logger = logging.getLogger('query_engine')

# === SYSTEM INSTRUCTION HANDLING ===
def get_system_instruction() -> str:
    """Load and prepare system instruction for query processing"""
    global _system_instruction
    
    if _system_instruction is not None:
        return _system_instruction
        
    # Load system instructions
    with open(SYSTEM_INSTRUCTION_FILE, "r") as f:
        _system_instruction = f.read()
    
    # Load filter values
    with open(FILTER_VALUES_FILE, "r") as f:
        all_filters = json.load(f)
        
    # Load data schema for simplified schema
    with open(DATA_SCHEMA_FILE, "r") as f:
        data_schema = json.load(f)
    
    # Replace the placeholders in system instruction
    _system_instruction = _system_instruction.replace("{all_filters}", json.dumps(all_filters))
    _system_instruction = _system_instruction.replace("{data_schema}", json.dumps(data_schema))
    
    return _system_instruction


# === DIMENSION HANDLING ===
def _get_dimension_types() -> Dict[str, str]:
    """Gets dimension types from data schema"""
    with open(DATA_SCHEMA_FILE, "r") as f:
        data_schema = json.load(f)
    
    dimension_types = {}
    for category in ["time_dimension", "geo_dimension", "categorical_dimension"]:
        for dim in data_schema["dimensions"].get(category, []):
            dimension_types[dim["physical_name"]] = dim["data_type"]
            
    return dimension_types


# === SQL GENERATION HELPERS ===
def _build_select_clause(definition: AggregationDefinition) -> str:
    """Builds the SELECT clause for a SQL query"""
    dims = definition.dimensions
    dims_clause = "\n  , ".join(dims) if dims else ""
    measures_clause = ", ".join([
        f"{m['expression']} AS {m['alias']}" 
        for m in definition.measures
    ])
    
    if dims_clause and measures_clause:
        return f"{dims_clause}\n  , {measures_clause}"
    else:
        return dims_clause or measures_clause


def _build_metadata_clauses(definition: AggregationDefinition) -> str:
    """Builds SQL clauses for metadata collection"""
    dims = definition.dimensions
    stats_metadata = ""
    
    # Add statistics for each measure
    for measure in definition.measures:
        measure_alias = measure['alias']
        
        # Add basic statistics as window functions
        stats_metadata += f"""
  , avg({measure_alias}) over () as metadata_avg_{measure_alias}
  , quantile_cont({measure_alias}, 0.25) over () as metadata_1Q_{measure_alias}
  , median({measure_alias}) over () as metadata_median_{measure_alias}
  , quantile_cont({measure_alias}, 0.75) over () as metadata_3Q_{measure_alias}
  , stddev(case when isinf({measure_alias}) then null else {measure_alias} end) over () as metadata_stddev_{measure_alias}"""
        
        # Add top/bottom entries with dimension values
        if dims:
            dim_array_elements = ", ".join(dims) + f", {measure_alias}"
            stats_metadata += f"""
  , cast(max_by(json_array({dim_array_elements}), {measure_alias}, 3) over () as string) as metadata_top_3_{measure_alias}
  , cast(min_by(json_array({dim_array_elements}), {measure_alias}, 3) over () as string) as metadata_bottom_3_{measure_alias}"""
    
    # Add result set size indicator
    stats_metadata += """
  , cast(sum(min(1)) over () > 5000 as string) as metadata_result_exceeds_limit"""
    
    # Add date range metadata
    date_metadata = """
  , min(min(created_date)) over () as metadata_min_created_date
  , max(max(created_date)) over () as metadata_max_created_date"""
    
    # Add location reference if needed
    location_reference = ""
    location_dims = ["location", "incident_zip", "neighborhood_name"]
    if dims and any(loc_dim in dim for dim in dims for loc_dim in location_dims):
        location_reference = """
  , min(borough) as reference_borough"""
        
        if dims and any("location" in dim for dim in dims):
            location_reference += """,
  string_agg(distinct nullif(neighborhood_name, 'Unspecified'), ', ') as reference_neighborhood"""
    
    return stats_metadata + date_metadata + location_reference


def _build_order_clause(definition: AggregationDefinition) -> str:
    """Builds the ORDER BY clause for a SQL query"""
    dims = definition.dimensions
    
    # Use topN if available
    if hasattr(definition, 'topN') and definition.topN:
        order_by_keys = ', '.join(definition.topN.orderByKey)
        return f"ORDER BY {order_by_keys}\nLIMIT {definition.topN.topN}"
    
    # Use standard ordering rules
    if len(dims) == 1:
        # Handle special cases for specific dimensions
        if dims[0] == 'time_to_resolve_day_bin':
            return "ORDER BY time_to_resolve_day_bin ASC"
        elif dims[0] == 'created_weekday_datepart':
            return "ORDER BY MIN(created_weekday_order) ASC"
        elif dims[0] == 'closed_weekday_datepart':
            return "ORDER BY MIN(closed_weekday_order) ASC"
        # Handle time dimensions
        elif dims[0] in TIME_DIMENSIONS:
            return f"ORDER BY {dims[0]} ASC"
        elif definition.measures:
            return f"ORDER BY {definition.measures[0]['alias']} DESC"
    elif definition.timeDimension:
        return f"ORDER BY {definition.timeDimension[0]} ASC"
    elif definition.measures:
        return f"ORDER BY {definition.measures[0]['alias']} DESC"
    
    return ""


def _extract_metadata_from_results(df: pd.DataFrame) -> Dict:
    """Extracts metadata from query results"""
    metadata = {}
    metadata_cols = [col for col in df.columns if col.startswith('metadata_')]
    
    if not metadata_cols:
        return metadata
    
    # Process date range metadata
    if 'metadata_min_created_date' in df.columns and 'metadata_max_created_date' in df.columns:
        metadata['createdDateRange'] = [
            df['metadata_min_created_date'].iloc[0],
            df['metadata_max_created_date'].iloc[0]
        ]
    
    # Process statistical metadata
    stats_metadata = {}
    for col in metadata_cols:
        if col.startswith('metadata_') and col not in ['metadata_min_created_date', 'metadata_max_created_date']:
            # Extract the statistic name and measure name
            parts = col.replace('metadata_', '').split('_', 1)
            if len(parts) == 2:
                stat_name, measure_name = parts
                
                if measure_name not in stats_metadata:
                    stats_metadata[measure_name] = {}
                
                # Store the statistic value
                stats_metadata[measure_name][stat_name] = df[col].iloc[0]
    
    # Add the statistics to the metadata
    if stats_metadata:
        metadata['statistics'] = stats_metadata
        
    return metadata


# === SQL GENERATION ===
def generate_sql(definition: AggregationDefinition, table_name: str) -> str:
    """
    Generates a SQL query from aggregation definition, keeping location placeholders intact.
    
    Args:
        definition: The aggregation definition
        table_name: The database table name
        
    Returns:
        SQL query string with placeholders for location data
    """
    start_time = time.time()
    logger.info("Generating SQL from aggregation definition")
        
    # Build SQL components
    select_clause = _build_select_clause(definition)
    metadata_clauses = _build_metadata_clauses(definition)
    full_select = f"{select_clause}{metadata_clauses}"
    
    # Create the SQL query
    sql = f"SELECT\n  {full_select}\nFROM {table_name}"
    
    # Create quality filters based on dimensions
    dimension_types = _get_dimension_types()
    dims = definition.dimensions
    quality_filters = []
    for dim in dims:
        if dim in dimension_types:
            data_type = dimension_types[dim]
            if data_type == "string":
                quality_filters.append(f"{dim} != 'Unspecified'")
            else:
                quality_filters.append(f"{dim} IS NOT NULL")
    
    # Add pre-aggregation filters if they exist
    where_clause_exists = False
    if definition.preAggregationFilters:
        sql += f"\nWHERE {definition.preAggregationFilters}"
        where_clause_exists = True
    
    # Add quality filters
    if quality_filters:
        quality_condition = " AND ".join(quality_filters)
        if where_clause_exists:
            sql += f" AND ({quality_condition})"
        else:
            sql += f"\nWHERE {quality_condition}"
    
    # Add grouping if dimensions exist
    if dims:
        group_clause = ", ".join(str(i) for i in range(1, len(dims) + 1))
        sql += f"\nGROUP BY {group_clause}"
    
    # Add post-aggregation filters (HAVING)
    if definition.postAggregationFilters:
        sql += f"\nHAVING {definition.postAggregationFilters}"
    
    # Add ordering
    order_clause = _build_order_clause(definition)
    if order_clause:
        sql += f"\n{order_clause}"
    
    # Add default limit if no ordering specified
    if "LIMIT" not in sql:
        sql += "\nLIMIT 5000"
    
    # Add semicolon
    if not sql.strip().endswith(";"):
        sql += ";"
    
    logger.info(f"SQL generation completed in {time.time() - start_time:.2f}s")
    
    return sql.strip()


# === SQL EXECUTION ===
def execute_sql_in_duckDB(
    sql: str, 
    db_filename: str, 
    user_location: Optional[Dict[str, float]] = None
) -> Tuple[List, Dict]:
    """
    Executes a SQL query in DuckDB, replacing location placeholders right before execution.
    
    Args:
        sql: SQL query with potential location placeholders
        db_filename: Path to DuckDB database file
        user_location: Optional user location data (lat/long)
        
    Returns:
        Tuple of (results, metadata)
    """
    start_time = time.time()
    logger.info("Executing SQL query in DuckDB")
    
    # Replace location placeholders only at execution time
    execution_sql = sql
    location_used = False
    
    if user_location and ("{{user_latitude}}" in sql or "{user_latitude}" in sql):
        location_used = True
        lat = user_location.get('latitude')
        lng = user_location.get('longitude')
        
        for placeholder, value in [
            ("{{user_latitude}}", str(lat)),
            ("{{user_longitude}}", str(lng)),
            ("{user_latitude}", str(lat)),
            ("{user_longitude}", str(lng))
        ]:
            execution_sql = execution_sql.replace(placeholder, value)
    
    metadata = {}
    try:
        with duckdb.connect(db_filename) as con:
            # Setup spatial extensions
            con.execute("INSTALL spatial;")
            con.execute("LOAD spatial;")
            con.execute("SET default_collation='nocase';")
            
            # Execute the query with substituted values
            df = con.execute(execution_sql).fetchdf()
            row_count = len(df)
            
            if location_used:
                logger.info(f"Location-based query executed successfully: {row_count} rows")
            else:
                logger.info(f"Query executed successfully: {row_count} rows")

            if row_count > 0:
                metadata = _extract_metadata_from_results(df)
                
                # Remove all metadata columns
                metadata_cols = [col for col in df.columns if col.startswith('metadata_')]
                if metadata_cols:
                    df = df.drop(columns=metadata_cols)
            else:
                logger.info("Query returned no results")
                
    except Exception as e:
        logger.error(f"Database query failed: {str(e)}")
        
        # Log sanitized SQL (without location data)
        sanitized_sql = execution_sql
        if location_used and user_location:
            # Replace actual coordinates with placeholders in error logs
            for coord in [str(user_location.get('latitude')), str(user_location.get('longitude'))]:
                sanitized_sql = sanitized_sql.replace(coord, "[MASKED_COORDINATE]")
                
        logger.error(f"SQL query with error:\n{sanitized_sql}")
        raise
    
    # Format datetime columns for JSON serialization
    for col in df.select_dtypes(include=['datetime64[ns]']).columns:
        df[col] = df[col].dt.strftime('%Y-%m-%d')
    
    # Format datetime values in metadata
    if 'createdDateRange' in metadata:
        metadata['createdDateRange'] = [
            value.strftime('%Y-%m-%d') if isinstance(value, pd.Timestamp) else value
            for value in metadata['createdDateRange']
        ]
    
    logger.info(f"SQL execution completed in {time.time() - start_time:.2f}s")
    results = json.loads(df.to_json(orient="records"))
    return results, metadata


# === QUERY PROCESSING ===
@gemini_safe
async def process_aggregation_query(
    translated_query: str,
    user_location: Optional[Dict[str, Any]],
    generate_content_safe
) -> Dict[str, Any]:
    """Process translated query text into a data query and execute it"""
    # Get system instruction
    system_instruction = get_system_instruction()
    
    # Use call_gemini_async directly instead of passing in generate_content_safe
    response = await call_gemini_async(
        "gemini-2.0-flash",
        translated_query,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction, 
            temperature=0
        )
    )
    
    # Parse the AI response using utility function
    json_text = response.candidates[0].content.parts[0].text
    parsed_json, is_valid_json = extract_json(json_text, return_status=True)
    
    # If not valid JSON or contains text response, return as text
    if not is_valid_json or "textResponse" in parsed_json:
        text = parsed_json.get("textResponse", json_text)
        return {
            "textResponse": text, 
            "chartType": "text", 
            "availableChartTypes": ["text"]
        }
    
    # Process data response - classify dimensions using utility function
    agg_def = AggregationDefinition(**parsed_json)
    time_dim, geo_dim, cat_dim = classify_dimensions(agg_def.dimensions)
    agg_def = agg_def.copy(update={
        "timeDimension": time_dim,
        "geoDimension": geo_dim,
        "categoricalDimension": cat_dim
    })
    
    # Generate SQL with placeholders intact
    sql = generate_sql(agg_def, "requests_311")
    
    # Check if location is required but not provided
    location_enabled = bool(user_location)
    if ('user_latitude' in sql or 'user_longitude' in sql) and not location_enabled:
        logger.info("Query requires location services but they are disabled")
        return {"locationRequired": True}
    
    # Execute SQL with location data
    dataset, query_metadata = execute_sql_in_duckDB(sql, DUCKDB_FILE, user_location)
    
    # Return basic query results
    return {
        "sql": sql,
        "dataset": dataset,
        "aggregationDefinition": agg_def,
        "queryMetadata": query_metadata
    }