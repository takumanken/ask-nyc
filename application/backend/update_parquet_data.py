from sodapy import Socrata
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import json
import time
import boto3
from botocore.client import Config
import duckdb
import tempfile
import shutil
import pandas as pd
import pytz

# Load environment variables
load_dotenv()
APP_TOKEN = os.environ.get("APP_TOKEN")

# Create temp directory for working with files
temp_dir = tempfile.mkdtemp()
print(f"Created temporary directory: {temp_dir}")

# File paths
PARQUET_OBJECT_NAME = 'requests_311.parquet'
GEO_OBJECT_NAME = '2020_nyc_neighborhood_tabulation_areas_nta.parquet'
temp_parquet_path = os.path.join(temp_dir, PARQUET_OBJECT_NAME)
temp_geo_path = os.path.join(temp_dir, GEO_OBJECT_NAME)
temp_updates_path = os.path.join(temp_dir, "recent_updates.json")
temp_updated_parquet_path = os.path.join(temp_dir, "updated_requests_311.parquet")

# Constants for date filtering
MIN_DATE = "2020-01-01T00:00:00.000"

def get_days_to_fetch():
    """Determine how many days to fetch based on current date"""
    ny_timezone = pytz.timezone('America/New_York')
    current_time_ny = datetime.now(ny_timezone)
    
    # Check if it's the 1st of the month
    if current_time_ny.day == 1:
        return 365, "annual refresh (1st of month)"
    
    # Check if it's Monday (weekday() returns 0 for Monday)
    elif current_time_ny.weekday() == 0:
        return 90, "weekly extended refresh (Monday)"
    
    # Default case
    else:
        return 14, "daily refresh"

def check_unique_ids(data_source, id_column_name, source_type):
    """Check if the given ID column contains only unique values"""
    try:
        print(f"Verifying unique IDs in {source_type}...")
        
        if source_type == "API data (JSON)":
            # For JSON, load and check
            with open(data_source, 'r') as f:
                data = json.load(f)
            
            total_count = len(data)
            unique_ids = set(item[id_column_name] for item in data)
            unique_count = len(unique_ids)
            
        elif source_type == "Existing data (Parquet)":
            # For Parquet, use DuckDB to query
            conn = duckdb.connect()
            total_count = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{data_source}')").fetchone()[0]
            unique_count = conn.execute(f"SELECT COUNT(DISTINCT \"{id_column_name}\") FROM read_parquet('{data_source}')").fetchone()[0]
            
            if total_count != unique_count:
                # Find duplicate IDs for reporting
                duplicates = conn.execute(f"""
                    SELECT "{id_column_name}", COUNT(*) as count 
                    FROM read_parquet('{data_source}')
                    GROUP BY "{id_column_name}"
                    HAVING COUNT(*) > 1
                    ORDER BY count DESC
                    LIMIT 10
                """).fetchall()
                
                print(f"WARNING: Found duplicate IDs in {source_type}. Examples:")
                for dup in duplicates:
                    print(f"  ID {dup[0]} appears {dup[1]} times")
        
        # Report findings
        if total_count == unique_count:
            print(f"✓ All IDs are unique in {source_type}: {total_count} records with {unique_count} unique IDs")
            return True
        else:
            print(f"⚠ WARNING: Duplicate IDs found in {source_type}: {total_count} records but only {unique_count} unique IDs")
            print(f"  {total_count - unique_count} duplicate records detected")
            return False
            
    except Exception as e:
        print(f"Error checking unique IDs in {source_type}: {str(e)}")
        return False

try:
    # Load R2 credentials from environment variables
    ACCOUNT_ID = os.environ.get('R2_ACCOUNT_ID')
    ACCESS_KEY_ID = os.environ.get('R2_ACCESS_KEY_ID')
    SECRET_ACCESS_KEY = os.environ.get('R2_SECRET_ACCESS_KEY')
    BUCKET_NAME = os.environ.get('R2_BUCKET_NAME')
    ENDPOINT_URL = f'https://{ACCOUNT_ID}.r2.cloudflarestorage.com'

    # Create R2 client
    session = boto3.session.Session()
    s3_client = session.client(
        's3',
        region_name='auto',
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=ACCESS_KEY_ID,
        aws_secret_access_key=SECRET_ACCESS_KEY,
        config=Config(signature_version='s3v4')
    )

    # 1. Determine how many days to fetch based on current date
    days_to_fetch, fetch_reason = get_days_to_fetch()
    
    # Calculate date range in New York time
    ny_timezone = pytz.timezone('America/New_York')
    current_time_ny = datetime.now(ny_timezone)
    start_date = (current_time_ny - timedelta(days=days_to_fetch)).strftime("%Y-%m-%dT00:00:00.000")
    
    print(f"Fetching records for {fetch_reason}")
    print(f"Date range: {start_date} to present ({days_to_fetch} days)")
    print(f"Records must also be created on or after {MIN_DATE}")
    
    # Set up Socrata client
    client = Socrata("data.cityofnewyork.us", app_token=APP_TOKEN, timeout=60)
    
    # Initialize file for updates
    with open(temp_updates_path, "w") as f:
        f.write("[\n")  # Start JSON array
        
        offset = 0
        count = 0
        first_record = True
        
        while True:
            start_time = time.time()
            print(f"  Fetching records {offset}-{offset+49999}...")
            
            results = client.get(
                "erm2-nwe9",
                where=f"created_date >= '{start_date}' AND created_date >= '{MIN_DATE}'",
                exclude_system_fields=False,
                limit=50000,
                offset=offset,
                order="unique_key"
            )
            
            if len(results) == 0:
                break
            
            # Write results to file as we go (streaming approach)
            for record in results:
                if not first_record:
                    f.write(",\n")
                else:
                    first_record = False
                
                json.dump(record, f)
                count += 1
            
            offset += 50000
            elapsed = time.time() - start_time
            print(f"  Fetched {len(results)} records in {elapsed:.2f} seconds")
            
            # Add a small delay to avoid hammering the API
            time.sleep(0.5)
            
        # Close the JSON array
        f.write("\n]")
    
    print(f"Completed fetching records: {count} total records saved to temporary file")
    
    # 2. Download the existing Parquet file from R2
    print(f"Downloading existing Parquet file from R2...")
    s3_client.download_file(BUCKET_NAME, PARQUET_OBJECT_NAME, temp_parquet_path)
    print(f"Downloaded existing Parquet file to {temp_parquet_path}")
    
    # Download geo file for spatial joins
    print(f"Downloading neighborhood data for spatial joins...")
    s3_client.download_file(BUCKET_NAME, GEO_OBJECT_NAME, temp_geo_path)
    print(f"Downloaded neighborhood data to {temp_geo_path}")
    
    # 3. Check for uniqueness in both datasets
    api_data_is_unique = check_unique_ids(temp_updates_path, "unique_key", "API data (JSON)")
    existing_data_is_unique = check_unique_ids(temp_parquet_path, "Unique Key", "Existing data (Parquet)")
    
    if not api_data_is_unique or not existing_data_is_unique:
        print("WARNING: Duplicate IDs detected. Proceeding with upsert but results may contain duplicates.")
    
    # 4. Perform upsert operation with DuckDB
    print("Performing upsert operation...")
    
    # Initialize DuckDB and load spatial extension
    conn = duckdb.connect()
    conn.execute("INSTALL spatial; LOAD spatial;")
    
    # Process the updates with spatial join in a single step
    conn.execute(f"""
        COPY (
            WITH updates AS (
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
                FROM read_json_auto('{temp_updates_path}') AS j
                LEFT JOIN read_parquet('{temp_geo_path}') AS n
                ON st_contains(
                    n.geometry,
                    st_point(CAST(j.longitude AS DOUBLE), CAST(j.latitude AS DOUBLE))
                )
            )
            
            SELECT * FROM (
                -- Keep records from the existing data that aren't being updated
                SELECT * FROM read_parquet('{temp_parquet_path}')
                WHERE "Unique Key" NOT IN (SELECT "Unique Key" FROM updates)
                
                UNION ALL
                
                -- Add all the updated records with their spatial join results
                SELECT * FROM updates
            )
        ) TO '{temp_updated_parquet_path}' (FORMAT 'parquet');
    """)
    
    # Get count of new records
    new_records_count = conn.execute(f"SELECT COUNT(*) FROM read_json_auto('{temp_updates_path}')").fetchone()[0]
    
    # Get total count
    total_count = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{temp_updated_parquet_path}')").fetchone()[0]
    
    # Final verification of the output file
    final_unique_count = conn.execute(f"SELECT COUNT(DISTINCT \"Unique Key\") FROM read_parquet('{temp_updated_parquet_path}')").fetchone()[0]
    if total_count != final_unique_count:
        print(f"⚠ WARNING: Final Parquet file contains duplicates: {total_count} records but only {final_unique_count} unique IDs")
    else:
        print(f"✓ Final Parquet file contains {total_count} records with no duplicates")
    
    print(f"Upsert complete. Added {new_records_count} new records. Total records: {total_count}")
    
    # 5. Upload the updated Parquet file back to R2
    print("Uploading updated Parquet file to R2...")
    s3_client.upload_file(temp_updated_parquet_path, BUCKET_NAME, PARQUET_OBJECT_NAME)
    print("Upload complete!")

except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()

finally:
    # Clean up temporary files
    print(f"Cleaning up temporary directory: {temp_dir}")
    shutil.rmtree(temp_dir)