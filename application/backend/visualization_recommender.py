"""
Visualization Recommender Module

This module recommends chart types based on data characteristics, 
including dimensionality, cardinality, and measure types.
"""

# Standard library imports
import logging
from typing import Dict, List, Tuple

# Local imports
from models import AggregationDefinition
from utils import classify_dimensions, TIME_DIMENSIONS, GEO_DIMENSIONS

# === GLOBAL CONFIGURATION ===
logger = logging.getLogger(__name__)

# === CONSTANTS ===
ADDITIVE_MEASURES = ["num_of_requests", "population"]
CARDINALITY_THRESHOLD = 15  # Maximum unique values for visualization without simplification

# === HELPER FUNCTIONS ===
def is_measure_additive(measure_alias: str) -> bool:
    # Determines if a measure can be summed (is additive).
    return measure_alias in ADDITIVE_MEASURES


def all_dimensions_exceed_cardinality(dimensions: List[str], dimension_stats: Dict[str, int], 
                                     threshold: int = CARDINALITY_THRESHOLD) -> bool:
    # Checks if all non-time dimensions exceed the cardinality threshold.
    non_time_dims = [dim for dim in dimensions if dim not in TIME_DIMENSIONS]

    # If no non-time dimensions, return False
    if len(non_time_dims) == 0:
        return False

    # Check cardinality for each non-time dimension
    # If any dimension's cardinality is less than or equal to the threshold, return False
    for dim in non_time_dims:
        cardinality = dimension_stats.get(dim, 0)
        if cardinality <= threshold:
            return False

    # If all non-time dimensions exceed the threshold, return True   
    return True


def is_topn_query(agg_def: AggregationDefinition) -> bool:
    
    # Determines if a query is a TOP N type query.
    return hasattr(agg_def, 'topN') and agg_def.topN is not None


# === VISUALIZATION RECOMMENDATION ===
def get_viz_recommendations(agg_def: AggregationDefinition, 
                          dimension_stats: Dict[str, int] = None, 
                          dataset_length: int = None) -> Tuple[List[str], str]:
    """
    Determines available and ideal chart types based on the aggregation definition.
    
    This is the core chart recommendation algorithm that analyzes dimension types,
    measure counts, cardinality, and other factors to recommend appropriate
    visualization types.
    
    Args:
        agg_def: The aggregation definition containing dimensions and measures
        dimension_stats: Statistics about dimensions' cardinality
        dataset_length: The number of rows in the result dataset
        
    Returns:
        Tuple of (available_charts, ideal_chart)
    """
    
    # Initialize defaults
    available = ["table"]  # Table is always available
    ideal = "table"        # Table is the default fallback

    # Extract query components
    dimensions = agg_def.dimensions or []
    measures = agg_def.measures or []

    # Classify dimensions by type
    time_dim, geo_dim, cat_dim = classify_dimensions(dimensions)
        
    # Count dimensions and measures by type
    dim_count = len(dimensions)
    time_count = len(time_dim)
    geo_count = len(geo_dim)
    cat_count = len(cat_dim)
    measure_count = len(measures)
    
    # Check measure additivity
    additive_measure_count = sum(is_measure_additive(m["alias"]) for m in measures) if measures else 0
    all_additive_measures = additive_measure_count == measure_count

    # Handle edge cases
    if dataset_length == 1:
        return available, ideal
    elif not dimensions and not measures:
        return available, ideal
    
    # Check if this is a TOP N query
    is_topn = is_topn_query(agg_def)
            
    # Check dimension cardinality
    if dimensions and dimension_stats:
        high_cardinality = all_dimensions_exceed_cardinality(dimensions, dimension_stats)
    
    # Check for location dimension (special case)
    is_not_location = dimensions[0] != "location" if dimensions else True
    
    # === VISUALIZATION SELECTION LOGIC ===
    
    # More than two dimensions or no measures -> default to table
    if dim_count > 2 or measure_count == 0:
        return available, ideal
    
    # Single categorical dimension with one measure (Bar Chart)
    if cat_count == 1 and measure_count == 1 and time_count == 0 and is_not_location:
        available.append("single_bar_chart")
        ideal = "single_bar_chart"
    
    # Time series data (Line Chart)
    if time_count == 1 and measure_count == 1 and not high_cardinality:
        available.append("line_chart")
        ideal = "line_chart"
        
        # Add stacked charts for categorical breakdowns of time series
        if cat_count == 1 and additive_measure_count == measure_count:
            available.append("stacked_area_chart")
            available.append("stacked_area_chart_100")
    
    # Multiple dimensions or measures (Nested Bar Chart)
    if 1 <= dim_count <= 2 and 1 <= measure_count <= 2 and (cat_count > 1 or measure_count > 1):
        available.append("nested_bar_chart")
        ideal = "nested_bar_chart"
    
    # Two categorical dimensions with one measure (Grouped Bar Chart)
    if cat_count == 2 and measure_count == 1 and not high_cardinality:
        available.append("grouped_bar_chart")
        ideal = "grouped_bar_chart"
        
        # Add stacked options if measures are additive
        if additive_measure_count == measure_count:
            available.append("stacked_bar_chart")
            available.append("stacked_bar_chart_100")
            ideal = "stacked_bar_chart"
    
    # Hierarchical data visualization (Treemap)
    if (1 <= dim_count <= 2 and measure_count == 1 and time_count == 0 
            and all_additive_measures and is_not_location and not is_topn):
        available.append("treemap")
    
    # Geospatial data visualization
    if geo_count == 1 and len(dimensions) == 1 and measure_count == 1:
        geo_name = geo_dim[0].lower()
        
        # Point data (Heatmap)
        if geo_name == "location":
            available.append("heatmap")
            available.remove("table")  # Remove table for point maps
            ideal = "heatmap"
            
        # Region data (Choropleth)
        elif geo_name in ["borough", "county", "neighborhood_name", "incident_zip"]:
            available.append("choropleth_map")
            ideal = "choropleth_map"
    
    logger.info(f"Chart options: {available}, ideal: {ideal}")
    return available, ideal