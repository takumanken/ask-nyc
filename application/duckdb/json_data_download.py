from sodapy import Socrata
from datetime import datetime
from dotenv import load_dotenv
import os
import json
import time

# Load environment variables
load_dotenv()
APP_TOKEN = os.environ.get("APP_TOKEN")

# Ensure data directories exist
os.makedirs("data/json", exist_ok=True)

# Setup Socrata client
client = Socrata("data.cityofnewyork.us", app_token=APP_TOKEN, timeout=60)

years = [2020, 2021, 2022, 2023, 2024, 2025]
for year in years:
    print(f"Processing data for {year}...")
    
    # Initialize file before loop to enable streaming
    with open(f"data/json/{year}.json", "w") as f:
        f.write("[\n")  # Start JSON array
        
        offset = 0
        count = 0
        first_record = True
        
        try:
            while True:
                start_time = time.time()
                print(f"  Fetching records {offset}-{offset+49999}...")
                
                results = client.get(
                    "erm2-nwe9",
                    where=f"created_date between '{year}-01-01T00:00:00' and '{year}-12-31T23:59:59'",
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
            
        except Exception as e:
            print(f"Error processing {year}: {str(e)}")
            # Close the JSON array even if we encounter an error
            f.write("\n]")
            continue
    
    print(f"Completed {year}: {count} total records saved to data/json/{year}.json")
    print("-" * 50)