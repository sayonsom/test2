import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import os
import psycopg2
from psycopg2.extras import RealDictCursor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConstructBreakdownEstimator:
    def __init__(self, production_columns: List[str]):
        """
        Initialize the Construct Breakdown estimator.
        
        Args:
            production_columns: List of production type columns (BAT, COL, GEO, etc.)
        """
        self.production_columns = production_columns
        
    def get_db_connection(self):
        """Get database connection"""
        return psycopg2.connect(
            dbname=os.environ.get('DB_NAME'),
            user=os.environ.get('DB_USER'),
            password=os.environ.get('DB_PASSWORD'),
            host=os.environ.get('DB_HOST'),
            port=os.environ.get('DB_PORT'),
            cursor_factory=RealDictCursor
        )
        
    def get_historical_data(self, current_time: datetime, region: str) -> pd.DataFrame:
        """
        Get historical data from the database, considering only data that would be
        available (i.e., data from previous months/years).
        
        Args:
            current_time: The timestamp we're estimating for
            region: The region to get data for
            
        Returns:
            DataFrame with historical data
        """
        # SQL query to get historical data
        query = """
        SELECT *
        FROM energy_metrics
        WHERE timestamp < DATE_TRUNC('month', %(current_time)s)
        AND region = %(region)s
        ORDER BY timestamp DESC
        """
        
        params = {
            'current_time': current_time,
            'region': region
        }
        
        try:
            with self.get_db_connection() as conn:
                historical_data = pd.read_sql_query(query, conn, params=params)
                
            if not historical_data.empty:
                # Convert timestamp to datetime
                historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'])
                # Calculate total production
                historical_data['total_production'] = historical_data[self.production_columns].sum(axis=1)
                
            return historical_data
            
        except Exception as e:
            logger.error(f"Error querying database for historical data: {e}")
            return pd.DataFrame()
        
    def estimate_total_production(self, 
                                timestamp: datetime,
                                region: str) -> float:
        """
        Estimate total production using historical monthly patterns.
        """
        # Get historical data from database
        historical_data = self.get_historical_data(timestamp, region)
        
        if historical_data.empty:
            logger.warning(f"No historical data found for region {region}")
            return 0.0
            
        # Get the month and hour for the timestamp
        month = timestamp.month
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Find relevant monthly aggregate from historical data
        monthly_data = historical_data[
            historical_data['timestamp'].dt.month == month
        ]
        
        if monthly_data.empty:
            # If no data for this month, use average of all months
            monthly_avg = historical_data['total_production'].mean()
            logger.info(f"Using overall average {monthly_avg} for {month}")
        else:
            monthly_avg = monthly_data['total_production'].mean()
            logger.info(f"Using monthly average {monthly_avg} for {month}")
            
        # Apply time-of-day factor (simple sinusoidal pattern)
        hour_factor = 1 + 0.2 * np.sin(2 * np.pi * (hour - 6) / 24)
        
        # Apply day-of-week factor (weekday vs weekend)
        day_factor = 0.9 if day_of_week >= 5 else 1.0  # Lower on weekends
        
        # Combine factors
        estimated_total = monthly_avg * hour_factor * day_factor
        
        return estimated_total
        
    def calculate_static_breakdown(self, 
                                timestamp: datetime,
                                region: str) -> Dict[str, float]:
        """
        Calculate static production mix breakdown from historical aggregates.
        """
        # Get historical data from database
        historical_data = self.get_historical_data(timestamp, region)
        
        if historical_data.empty:
            logger.warning(f"No historical data found for region {region}")
            return {col: 0.0 for col in self.production_columns}
            
        # Calculate average production for each type
        total_production = historical_data[self.production_columns].sum().sum()
        
        if total_production == 0:
            logger.warning(f"Zero total production found for region {region}")
            return {col: 0.0 for col in self.production_columns}
            
        breakdown = {}
        for col in self.production_columns:
            breakdown[col] = historical_data[col].sum() / total_production
            
        return breakdown
        
    def estimate_production(self,
                          timestamp: datetime,
                          region: str) -> Dict[str, float]:
        """
        Main estimation method that combines total production estimation
        with static breakdown to get final production mix.
        
        Returns dictionary with estimated values for each production type.
        """
        # First estimate total production
        total_production = self.estimate_total_production(
            timestamp, region
        )
        
        # Get static breakdown
        breakdown = self.calculate_static_breakdown(
            timestamp, region
        )
        
        # Apply breakdown to total production
        estimates = {}
        for prod_type, percentage in breakdown.items():
            estimates[prod_type] = total_production * percentage
            
        return estimates
        
    def fill_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing data in the dataframe using the Construct Breakdown method.
        Handles cases where the latest data is months behind.
        """
        result_df = df.copy()
        
        # Identify missing data (where updatedAt != timestamp)
        missing_mask = df['timestamp'] != df['updatedAt']
        missing_rows = df[missing_mask]
        
        logger.info(f"Found {len(missing_rows)} rows with missing data")
        
        for _, row in missing_rows.iterrows():
            # Get total production estimate
            total_production = self.estimate_total_production(
                row['timestamp'],
                row['region']
            )
            
            # Get production mix breakdown
            breakdown = self.calculate_static_breakdown(
                row['timestamp'],
                row['region']
            )
            
            # Calculate estimates
            estimates = {
                col: total_production * percentage 
                for col, percentage in breakdown.items()
            }
            
            # Update the result dataframe
            result_df.loc[
                (result_df['timestamp'] == row['timestamp']) &
                (result_df['region'] == row['region']),
                list(estimates.keys())
            ] = list(estimates.values())
            
            logger.info(f"Filled data for {row['timestamp']} in region {row['region']}")
        
        return result_df

# Example usage:
if __name__ == "__main__":
    # Example production columns
    production_columns = ['BAT', 'COL', 'GEO', 'NG', 'NUC', 'OES', 'OIL', 
                         'OTH', 'PS', 'SNB', 'SUN', 'UES', 'WAT', 'WND']
    
    # Initialize estimator
    estimator = ConstructBreakdownEstimator(production_columns)
    
    # Example current data with missing values
    example_json = {
        "data": [
            {
                "timestamp": "2024-02-24T02:00:00",
                "updatedAt": "2023-11-24T09:00:00",  # Several months behind
                "region": "Zone1",
                "BAT": 25.5,
                "COL": 30.2,
                "GEO": 15.7,
                "NG": 45.3,
                "NUC": 92.1,
                "OES": 28.4,
                "OIL": 22.9,
                "OTH": 33.6,
                "PS": 27.8,
                "SNB": 18.4,
                "SUN": 45.2,
                "UES": 31.5,
                "WAT": 40.2,
                "WND": 25.8
            }
        ]
    }
    
    # Convert JSON to DataFrame
    df = pd.DataFrame(example_json["data"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['updatedAt'] = pd.to_datetime(df['updatedAt'])
    
    # Fill the missing data
    filled_df = estimator.fill_missing_data(df)
    print("Data filled successfully!")
