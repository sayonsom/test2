import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import os
import psycopg2
from psycopg2.extras import RealDictCursor

class TSAEstimator:
    def __init__(self, columns_to_estimate: List[str]):
        """
        Initialize the TSA estimator with the columns that need to be estimated.
        
        Args:
            columns_to_estimate: List of column names to apply the estimation to
        """
        self.columns_to_estimate = columns_to_estimate
        
    def identify_gaps(self, df: pd.DataFrame) -> List[Dict]:
        """
        Identify gaps where timestamp and updatedAt hours don't match.
        
        Args:
            df: DataFrame with timestamp and updatedAt columns
            
        Returns:
            List of dictionaries containing gap information
        """
        gaps = []
        df = df.sort_values('timestamp')
        
        # Convert timestamps to datetime if they aren't already
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['updatedAt'] = pd.to_datetime(df['updatedAt'])
        
        # Find rows where hours don't match
        mask = df['timestamp'].dt.hour != df['updatedAt'].dt.hour
        gap_rows = df[mask]
        
        for _, row in gap_rows.iterrows():
            gaps.append({
                'start_time': row['timestamp'],
                'end_time': row['updatedAt'],
                'region': row['region']
            })
            
        return gaps
    
    def get_similar_periods(self, df: pd.DataFrame, gap: Dict, window_size: int = 1) -> pd.DataFrame:
        """
        Get data points from the same time period on different days in the same month by querying the database.
        
        Args:
            df: DataFrame with connection info (not used directly anymore)
            gap: Dictionary containing gap information
            window_size: Hours before and after the gap time to consider
            
        Returns:
            DataFrame containing similar periods
        """
        gap_start = gap['start_time']
        gap_hour = gap_start.hour
        
        # SQL query to get similar periods
        query = """
        SELECT *
        FROM energy_data
        WHERE EXTRACT(MONTH FROM timestamp) = EXTRACT(MONTH FROM %(gap_start)s)
        AND EXTRACT(YEAR FROM timestamp) = EXTRACT(YEAR FROM %(gap_start)s)
        AND region = %(region)s
        AND EXTRACT(HOUR FROM timestamp) BETWEEN %(min_hour)s AND %(max_hour)s
        AND DATE(timestamp) != DATE(%(gap_start)s)
        """
        
        params = {
            'gap_start': gap_start,
            'region': gap['region'],
            'min_hour': gap_hour - window_size,
            'max_hour': gap_hour + window_size
        }
        
        try:
            # TODO:  Integrate the DB connection here. 
            # Also,we can try to use a cache here to avoid querying the database every time
            with get_db_connection() as conn:
                similar_periods = pd.read_sql_query(query, conn, params=params)
                
            return similar_periods
            
        except Exception as e:
            print(f"Error querying database for similar periods: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error
    
    def estimate_gap_values(self, df: pd.DataFrame, gap: Dict) -> pd.DataFrame:
        """
        Estimate values for a gap using the TSA method.
        
        Args:
            df: Complete DataFrame
            gap: Dictionary containing gap information
            
        Returns:
            DataFrame with estimated values
        """
        similar_periods = self.get_similar_periods(df, gap)
        
        if similar_periods.empty:
            return pd.DataFrame()  # Return empty DataFrame if no similar periods found
            
        # Calculate average values for each column
        estimated_values = {}
        for col in self.columns_to_estimate:
            estimated_values[col] = similar_periods[col].mean()
            
        # Create a new row with estimated values
        estimated_row = {
            'timestamp': gap['start_time'],
            'updatedAt': gap['start_time'],  # Use the same timestamp to indicate it's estimated
            'region': gap['region'],
            **estimated_values
        }
        
        return pd.DataFrame([estimated_row])
    
    def align_boundaries(self, original_data: pd.DataFrame, estimated_data: pd.DataFrame) -> pd.DataFrame:
        """
        Align the estimated values to ensure continuity at gap boundaries.
        
        Args:
            original_data: Original DataFrame
            estimated_data: DataFrame with estimated values
            
        Returns:
            DataFrame with aligned estimated values
        """
        aligned_data = estimated_data.copy()
        
        # Find the values before and after the gap
        for col in self.columns_to_estimate:
            start_value = original_data[col].iloc[0]
            end_value = original_data[col].iloc[-1]
            
            # Linear interpolation between boundaries
            n_points = len(aligned_data)
            slope = (end_value - start_value) / (n_points + 1)
            
            for i in range(n_points):
                aligned_data.loc[aligned_data.index[i], col] = start_value + slope * (i + 1)
                
        return aligned_data
    
    def fill_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main method to fill gaps in the data using TSA estimation.
        
        Args:
            df: Input DataFrame with potential gaps
            
        Returns:
            DataFrame with filled gaps
        """
        result_df = df.copy()
        gaps = self.identify_gaps(df)
        
        for gap in gaps:
            # Get the data right before and after the gap for alignment
            gap_context = df[
                (df['timestamp'] >= gap['start_time'] - timedelta(hours=1)) &
                (df['timestamp'] <= gap['end_time'] + timedelta(hours=1)) &
                (df['region'] == gap['region'])
            ]
            
            # Estimate values for the gap
            estimated_values = self.estimate_gap_values(df, gap)
            if not estimated_values.empty:
                # Align the estimates with the boundaries
                aligned_estimates = self.align_boundaries(gap_context, estimated_values)
                
                # Add the estimates to the result
                result_df = pd.concat([result_df, aligned_estimates], ignore_index=True)
        
        # Sort the final result by timestamp and region
        result_df = result_df.sort_values(['timestamp', 'region'])
        
        return result_df

def get_db_connection():
    """Get the AWS Connection here"""
    return psycopg2.connect(
        dbname=os.environ.get('DB_NAME'),
        user=os.environ.get('DB_USER'),
        password=os.environ.get('DB_PASSWORD'),
        host=os.environ.get('DB_HOST'),
        port=os.environ.get('DB_PORT'),
        cursor_factory=RealDictCursor
    )

# Just for demo testing today in the terminal
if __name__ == "__main__":
    # Example columns that need to be estimated
    columns_to_estimate = ['BAT', 'COL', 'GEO', 'NG', 'NUC', 'OES', 'OIL', 
                          'OTH', 'PS', 'SNB', 'SUN', 'UES', 'WAT', 'WND']
    
    # Initialize the estimator
    estimator = TSAEstimator(columns_to_estimate)
    
    # Example JSON input (in practice, this would come from our Postgres database in AWS)
    example_json = {
        "data": [
            {
                "timestamp": "2024-02-24T02:00:00",
                "updatedAt": "2024-02-24T09:00:00",
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
            },
            {
                "timestamp": "2024-02-24T03:00:00",
                "updatedAt": "2024-02-24T09:00:00",
                "region": "Zone1",
                "BAT": 24.5,
                "COL": 20.2,
                "GEO": 15.7,
                "NG": 45.3,
                "NUC": 92.1,
                "OES": 28.4,
                "OIL": 22.9,
                
            },
            # we can try adding more data points to see how the estimation works
        ]
    }
    
    # Convert JSON to DataFrame
    df = pd.DataFrame(example_json["data"])
    
    # Convert timestamp columns to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['updatedAt'] = pd.to_datetime(df['updatedAt'])
    
    # Check if there are any gaps
    gaps = estimator.identify_gaps(df)
    
    if gaps:
        print(f"Found {len(gaps)} gaps in the data. Proceeding with estimation...")
        # Fill the gaps
        filled_df = estimator.fill_gaps(df)
        
        # Convert back to JSON format
        result_json = {
            "data": filled_df.to_dict(orient='records')
        }
        print("Gaps filled successfully.")
        print(result_json)
    else:
        print("No gaps found in the data.")
        result_json = example_json
    
    # The result_json can now be returned or saved as needed