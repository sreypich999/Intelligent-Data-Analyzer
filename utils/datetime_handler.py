import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DatetimeHandler:
    """
    Utility class for handling datetime conversions and extractions in DataFrames.
    """
    def __init__(self):
        logging.info("DatetimeHandler initialized.")

    def convert_to_datetime(self, df: pd.DataFrame, column: str, errors: str = 'coerce') -> pd.DataFrame:
        """
        Converts a specified column to datetime objects.
        'errors' can be 'ignore', 'raise', or 'coerce'.
        """
        if column not in df.columns:
            logging.warning(f"Column '{column}' not found in DataFrame for datetime conversion.")
            return df
        
        original_dtype = df[column].dtype
        df[column] = pd.to_datetime(df[column], errors=errors)
        
        if df[column].dtype == original_dtype:
            logging.warning(f"Column '{column}' could not be converted to datetime. Data type remains {original_dtype}.")
        else:
            logging.info(f"Column '{column}' successfully converted to datetime. Original: {original_dtype}, New: {df[column].dtype}")
        return df

    def extract_datetime_features(self, df: pd.DataFrame, column: str, features: list = None) -> pd.DataFrame:
        """
        Extracts various datetime features (year, month, day, hour, etc.) from a datetime column.
        If features is None, extracts common ones.
        """
        if column not in df.columns or not pd.api.types.is_datetime64_any_dtype(df[column]):
            logging.warning(f"Column '{column}' is not a datetime type or not found. Cannot extract features.")
            return df

        if features is None:
            features = ['year', 'month', 'day', 'hour', 'minute', 'dayofweek', 'dayofyear', 'weekofyear', 'quarter']

        for feature in features:
            new_col_name = f"{column}_{feature}"
            if feature == 'year':
                df[new_col_name] = df[column].dt.year
            elif feature == 'month':
                df[new_col_name] = df[column].dt.month
            elif feature == 'day':
                df[new_col_name] = df[column].dt.day
            elif feature == 'hour':
                df[new_col_name] = df[column].dt.hour
            elif feature == 'minute':
                df[new_col_name] = df[column].dt.minute
            elif feature == 'second':
                df[new_col_name] = df[column].dt.second
            elif feature == 'dayofweek': # Monday=0, Sunday=6
                df[new_col_name] = df[column].dt.dayofweek
            elif feature == 'dayofyear':
                df[new_col_name] = df[column].dt.dayofyear
            elif feature == 'weekofyear':
                # Use .dt.isocalendar().week which is more robust than .dt.weekofyear (deprecated)
                df[new_col_name] = df[column].dt.isocalendar().week.astype(int) 
            elif feature == 'quarter':
                df[new_col_name] = df[column].dt.quarter
            else:
                logging.warning(f"Unknown datetime feature '{feature}' requested for extraction.")
        
        logging.info(f"Extracted features {', '.join(features)} from column '{column}'.")
        return df