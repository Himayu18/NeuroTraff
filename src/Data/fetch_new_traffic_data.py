import sys
import os
from datetime import datetime
from src.components.data_ingestion import DataIngestion
from src.logger import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))




REFERENCE_DATE = datetime(2025, 9, 27)

def should_run_today(reference_date: datetime) -> bool:
    today = datetime.today().date()
    delta_days = (today - reference_date.date()).days
    return delta_days > 0 and delta_days % 3 == 0

if __name__ == "__main__":
    if not should_run_today(REFERENCE_DATE):
        logging.info("Skipping run: Not the 3rd day.")
        sys.exit(0)

    logging.info("Fetching data from MongoDB...")
    fetch_new_traffic_data = DataIngestion()
    fetch_new_traffic_data.fetch_new_data()
