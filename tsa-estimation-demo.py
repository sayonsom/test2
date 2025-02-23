import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import matplotlib.pyplot as plt

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
        Get data points from the same time period on different days in the same month.
        
        Args:
            df: Complete DataFrame
            gap: Dictionary containing gap information
            window_size: Hours before and after the gap time to consider
            
        Returns:
            DataFrame containing similar periods
        """
        gap_start = gap['start_time']
        gap_hour = gap_start.hour
        
        # Filter data for the same month and region
        month_data = df[
            (df['timestamp'].dt.month == gap_start.month) &
            (df['timestamp'].dt.year == gap_start.year) &
            (df['region'] == gap['region'])
        ]
        
        # Get data points from the same time period on different days
        similar_periods = month_data[
            (month_data['timestamp'].dt.hour >= (gap_hour - window_size)) &
            (month_data['timestamp'].dt.hour <= (gap_hour + window_size)) &
            (month_data['timestamp'].dt.date != gap_start.date())
        ]
        
        return similar_periods
    
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

# Example usage:
if __name__ == "__main__":
    # Add visualization flag
    show_plots = True
    
    # Example columns that need to be estimated
    columns_to_estimate = ['BAT', 'COL', 'GEO', 'NG', 'NUC', 'OES', 'OIL', 
                          'OTH', 'PS', 'SNB', 'SUN', 'UES', 'WAT', 'WND']
    
    # Initialize the estimator
    estimator = TSAEstimator(columns_to_estimate)
    
    # Example DataFrame creation (you would load your actual data here)
    example_data = {
        'timestamp': pd.date_range(start='2024-02-24', periods=24, freq='H'),
        'updatedAt': pd.date_range(start='2024-02-24', periods=24, freq='H'),
        'region': ['Zone1'] * 24,
    }
    
    # I am just putting some example values for each column with a more realistic pattern
    np.random.seed(2025)  # Just some random number to make the data more realistic
    base_values = {
        'SUN': np.concatenate([np.zeros(6), np.linspace(0, 100, 12), np.zeros(6)]),  # Solar pattern
        'WND': 20 + 10 * np.sin(np.linspace(0, 4*np.pi, 24)),  # Wind pattern
        'NG': 40 + np.random.rand(24) * 10,  # Natural gas baseload
        'NUC': 90 + np.random.rand(24) * 5,  # Nuclear baseload
    }
    
    for col in columns_to_estimate:
        if col in base_values:
            example_data[col] = base_values[col]
        else:
            example_data[col] = 30 + np.random.rand(24) * 20
            
    df = pd.DataFrame(example_data)

    print("Original DataFrame:")
    print(df)
    
    # Introduce a gap by keeping timestamp stuck at 9 AM
    mask = (df['timestamp'].dt.hour >= 10) & (df['timestamp'].dt.hour <= 15)
    reference_time = df[df['timestamp'].dt.hour == 9]['timestamp'].iloc[0]

    # Keep the timestamp stuck at 9 AM for the gap period
    df.loc[mask, 'timestamp'] = reference_time

    # Keep the corresponding data columns stuck at 9 AM values
    reference_data = df[df['timestamp'].dt.hour == 9].iloc[0]
    for col in columns_to_estimate:
        df.loc[mask, col] = reference_data[col]
    
    print("\nAfter introducing the gap:")
    print(df)

    # Fill the gaps
    filled_df = estimator.fill_gaps(df)
    print("\nAfter filling the gaps:")
    print(filled_df)

    if show_plots:
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('TSA Estimation Example - Selected Energy Sources')
        
        # Plot interesting columns
        plot_columns = ['SUN', 'WND', 'NG', 'NUC']
        for ax, col in zip(axes.flat, plot_columns):
            # Original data
            original_data = example_data['timestamp'].values
            ax.plot(original_data, example_data[col], 
                   label='Original', marker='o', linestyle='-', alpha=0.5)
            
            # Data with gap
            gap_data = df.sort_values('updatedAt')
            ax.plot(gap_data['updatedAt'], gap_data[col], 
                   label='With Gap', marker='x', linestyle='--', alpha=0.5)
            
            # Filled data
            filled_data = filled_df.sort_values('updatedAt')
            ax.plot(filled_data['updatedAt'], filled_data[col], 
                   label='Filled', marker='^', linestyle=':', alpha=0.7)
            
            ax.set_title(f'{col} Values Over Time')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()