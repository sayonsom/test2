#!/usr/bin/env python3 
import requests 
from datetime import datetime
import pytz
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def costaRica(): 
    url = "https://apps.grupoice.com/CenceWeb/data/sen/json/EnergiaHorariaFuentePlanta"
    try:
        response = requests.get(url)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return None

    data_json = response.json()
    records = data_json.get("data", [])
    if not records:
        logger.warning("No data found in the JSON response.")
        return None

    # Convert the "fecha" field into a datetime object for each record
    for rec in records:
        try:
            rec["dt"] = datetime.strptime(rec["fecha"], "%Y-%m-%d %H:%M:%S.%f")
        except Exception:
            rec["dt"] = None

    # Find the latest available hour
    valid_times = [rec["dt"] for rec in records if rec["dt"] is not None]
    if not valid_times:
        logger.warning("No valid datetime found in records.")
        return None
    latest_time = max(valid_times)

    # Filter to only records for the latest hour
    latest_records = [rec for rec in records if rec.get("dt") == latest_time]

    # Initialize all possible keys with 0
    production = {
        "BIO": 0.0,  # biomass
        "BAT": 0.0,  # battery storage
        "COL": 0.0,  # coal
        "GEO": 0.0,  # geothermal
        "NG": 0.0,   # natural gas
        "NUC": 0.0,  # nuclear
        "OIL": 0.0,  # oil
        "OTH": 0.0,  # other
        "SNS": 0.0,  # solar (specific subset)
        "SUN": 0.0,  # standard solar
        "UES": 0.0,  # UESpecified
        "WAT": 0.0,  # water/hydroelectric
        "WND": 0.0,  # wind
    }

    # Map Costa Rica's energy types to our standardized keys
    source_mapping = {       # Water/Hydro
        "Bagazo": "BIO",
        "Eólica": "WND",
        "Geotérmica": "GEO",
        "Solar": "SUN",
        "Térmica": "OIL",
        "Hidroeléctrica": "WAT"
    }


    # Aggregate production by energy type
    for rec in latest_records:
        source = rec.get("fuente", "Unknown")
        value = rec.get("dato", 0)
        mapped_source = source_mapping.get(source, "UES")  # Default to UES if unknown
        production[mapped_source] += float(value)
        logger.debug(f"Mapped {source} to {mapped_source}: {value}")

    # Build the final dictionary
    output = {
        "region": "costaRica",
        "updatedAt": datetime.now(pytz.UTC).isoformat(),
        "timestamp": latest_time.replace(tzinfo=pytz.UTC).isoformat(),
    }
    output.update(production)

    logger.info(f"Successfully processed data for timestamp: {output['timestamp']}")
    return output

if __name__ == "__main__": 
    energy_data = costaRica() 
    if energy_data: 
        logger.info("National Energy Production Breakdown:") 
        logger.info(energy_data) 
    else: 
        logger.error("Failed to obtain energy data.")