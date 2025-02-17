import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import logging
import json
from typing import Dict, Union, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cyprus_power.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CyprusPowerScraper:
    def __init__(self):
        self.url = "https://tsoc.org.cy/electrical-system/total-daily-system-generation-on-the-transmission-system/"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    def fetch_data(self) -> Optional[str]:
        """Fetch HTML data from the TSOC website."""
        try:
            logger.debug("Initiating request to TSOC website")
            response = requests.get(self.url, headers=self.headers)
            response.raise_for_status()
            logger.info("Successfully fetched data from TSOC website")
            return response.text
        except requests.RequestException as e:
            logger.error(f"Failed to fetch data: {str(e)}")
            return None

    def parse_latest_data(self, html: str) -> Optional[Dict[str, Union[str, float]]]:
        """Parse the HTML and extract the latest hour's data."""
        try:
            logger.debug("Parsing HTML content")
            soup = BeautifulSoup(html, 'html.parser')
            table = soup.find('table', id='production_graph_data')
            
            if not table:
                logger.error("Table not found in HTML content")
                return None

            # Find the latest row with ':00:00' timestamp
            rows = table.find('tbody').find_all('tr')
            latest_hour_row = None
            
            for row in rows:
                timestamp = row.find('td').text.strip()
                if timestamp.endswith(':00:00'):
                    latest_hour_row = row
                    break

            if not latest_hour_row:
                logger.error("No hourly data found")
                return None

            # Extract data from the row
            cells = latest_hour_row.find_all('td')
            timestamp = cells[0].text.strip()
            
            # Map the data to the required format
            data = {
                "region": "cyprus",
                "updatedAt": datetime.now().isoformat(),
                "BAT": 0.0,  # Battery storage (not present)
                "COL": float(cells[5].text.strip()),  # Conventional Generation
                "GEO": 0.0,  # Geothermal (not present)
                "NG": 0.0,  # Natural gas (not present)
                "NUC": 0.0,  # Nuclear (not present)
                "OES": 0.0,  # Other energy sources
                "OIL": 0.0,  # Oil
                "OTH": 0.0,  # Other
                "PS": 0.0,  # Pumped storage (not present)
                "SNB": 0.0,  # Solar new baseline (not present)
                "SUN": float(cells[3].text.strip()),  # Distributed Generation (mostly solar)
                "UES": 0.0,  # Unspecified
                "WAT": 0.0,  # Hydroelectric (not present)
                "WND": float(cells[2].text.strip())  # Wind Generation
            }

            logger.info(f"Successfully parsed data for timestamp: {timestamp}")
            return data

        except Exception as e:
            logger.error(f"Error parsing data: {str(e)}")
            return None

    def save_to_json(self, data: Dict[str, Union[str, float]], filename: str = "cyprus_power.json"):
        """Save the data to a JSON file."""
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)
            logger.info(f"Successfully saved data to {filename}")
        except Exception as e:
            logger.error(f"Error saving to JSON: {str(e)}")

def main():
    """Main function to run the scraper."""
    logger.info("Starting Cyprus power data collection")
    
    scraper = CyprusPowerScraper()
    html_content = scraper.fetch_data()
    
    if html_content:
        data = scraper.parse_latest_data(html_content)
        if data:
            scraper.save_to_json(data)
            logger.info("Data collection completed successfully")
        else:
            logger.error("Failed to parse data")
    else:
        logger.error("Failed to fetch HTML content")

if __name__ == "__main__":
    main()