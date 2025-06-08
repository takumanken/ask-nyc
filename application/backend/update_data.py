from sodapy import Socrata
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import os

load_dotenv()
APP_TOKEN = os.environ.get("APP_TOKEN")

client = Socrata("data.cityofnewyork.us",
                 app_token=APP_TOKEN, timeout=60)

yesterday = (datetime.now(timezone.utc)
             - timedelta(days=1)).strftime("%Y-%m-%dT04:00:00")

results = client.get(
    "erm2-nwe9",
    where=f":updated_at >= '{yesterday}'",
    exclude_system_fields=False,
    limit=5
)

print(results)