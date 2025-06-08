import duckdb
import glob
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import logging
import time
import os

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])

logging.info("Starting JSON to Parquet conversion script.")

try:
    # Create output directories if they don't exist
    os.makedirs("data/geo", exist_ok=True)
    os.makedirs("data/parquet", exist_ok=True)
    
    # File paths
    geojson_path = "2020_nyc_neighborhood_tabulation_areas_nta_raw.geojson"
    parquet_geo_path = "data/geo/2020_nyc_neighborhood_tabulation_areas_nta.parquet"
    json_pattern = 'data/json/*.json'
    output_parquet_path = 'data/parquet/requests_311.parquet'

    logging.info(f"Loading GeoJSON file: {geojson_path}")
    start_time = time.time()
    nta = gpd.read_file(geojson_path)
    logging.info(f"GeoJSON loaded successfully. Shape: {nta.shape}. Time taken: {time.time() - start_time:.2f} seconds.")

    logging.info("Converting GeoDataFrame CRS to EPSG:4326.")
    nta = nta.to_crs("EPSG:4326")
    logging.info("CRS conversion complete.")

    # Save the GeoJSON as a Parquet file
    logging.info(f"Saving GeoDataFrame to Parquet file: {parquet_geo_path}")
    start_time = time.time()
    nta.to_parquet(parquet_geo_path)
    logging.info(f"GeoDataFrame saved to Parquet. Time taken: {time.time() - start_time:.2f} seconds.")

    # Load the spatial extension
    logging.info("Initializing DuckDB and loading spatial extension.")
    start_time = time.time()
    duckdb.sql("INSTALL spatial; LOAD spatial;")
    logging.info(f"DuckDB spatial extension loaded. Time taken: {time.time() - start_time:.2f} seconds.")

    # Construct and execute the main SQL query for JSON files with proper type casting
    sql_query = f"""
        COPY (
            SELECT 
                CAST(j.unique_key AS BIGINT) AS "Unique Key",
                CAST(j.created_date AS TIMESTAMP) AS "Created Date",
                CAST(j.closed_date AS TIMESTAMP) AS "Closed Date",
                j.agency AS "Agency",
                j.agency_name AS "Agency Name",
                j.complaint_type AS "Complaint Type",
                j.descriptor AS "Descriptor",
                j.location_type AS "Location Type",
                j.incident_zip AS "Incident Zip",
                j.incident_address AS "Incident Address",
                j.street_name AS "Street Name",
                j.cross_street_1 AS "Cross Street 1",
                j.cross_street_2 AS "Cross Street 2",
                j.intersection_street_1 AS "Intersection Street 1",
                j.intersection_street_2 AS "Intersection Street 2",
                j.address_type AS "Address Type",
                j.city AS "City",
                j.landmark AS "Landmark",
                j.status AS "Status",
                j.resolution_description AS "Resolution Description",
                CAST(j.resolution_action_updated_date AS TIMESTAMP) AS "Resolution Action Updated Date",
                j.community_board AS "Community Board",
                j.bbl AS "BBL",
                j.borough AS "Borough",
                CAST(j.x_coordinate_state_plane AS BIGINT) AS "X Coordinate (State Plane)",
                CAST(j.y_coordinate_state_plane AS BIGINT) AS "Y Coordinate (State Plane)",
                j.open_data_channel_type AS "Open Data Channel Type",
                j.park_facility_name AS "Park Facility Name",
                j.park_borough AS "Park Borough",
                CAST(j.latitude AS DOUBLE) AS "Latitude",
                CAST(j.longitude AS DOUBLE) AS "Longitude",
                j.":created_at" AS "Record Created At",
                j.":updated_at" AS "Record Updated At",
                n.nta2020 AS neighborhood_code,
                n.ntaname AS neighborhood_name,
                n.shape_area AS neighborhood_area,
                n.ntatype AS neighborhood_type
            FROM read_json_auto('{json_pattern}') AS j
            LEFT JOIN read_parquet('{parquet_geo_path}') AS n
              ON st_contains(
                  n.geometry,
                  st_point(CAST(j.longitude AS DOUBLE), CAST(j.latitude AS DOUBLE))
              )
        ) TO '{output_parquet_path}' (FORMAT 'parquet');
    """
    logging.info(f"Executing DuckDB SQL query with type casting...")
    logging.debug(f"SQL Query:\n{sql_query}")
    start_time = time.time()
    duckdb.sql(sql_query)
    logging.info(f"DuckDB query executed successfully. Output saved to {output_parquet_path}. Time taken: {time.time() - start_time:.2f} seconds.")

except Exception as e:
    logging.error(f"An error occurred during the conversion process: {e}", exc_info=True)

finally:
    logging.info("Script finished.")