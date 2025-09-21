import pandas as pd
import numpy as np
import logging
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataCleaner:
    """Enhanced data cleaner with advanced imputation strategies"""
    def __init__(self):
        logging.info("DataCleaner initialized")

    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'auto', fill_value=None) -> pd.DataFrame:
        """
        Enhanced missing value handling with automatic strategy selection
        Strategies: 'auto', 'drop', 'mean', 'median', 'mode', 'knn', 'constant'
        """
        df_cleaned = df.copy()
        
        # Auto-detect best strategy
        if strategy == 'auto':
            num_missing = df_cleaned.isnull().sum().sum()
            total_cells = df_cleaned.size
            missing_percent = num_missing / total_cells
            
            if missing_percent < 0.05:
                strategy = 'drop'
            elif missing_percent < 0.3:
                strategy = 'knn'
            else:
                strategy = 'median'
                
            logging.info(f"Auto-selected strategy: {strategy}")

        # Apply selected strategy
        if strategy == 'drop':
            original_rows = df_cleaned.shape[0]
            df_cleaned.dropna(inplace=True)
            logging.info(f"Dropped {original_rows - df_cleaned.shape[0]} rows")
        elif strategy == 'knn':
            imputer = KNNImputer(n_neighbors=5)
            num_cols = df_cleaned.select_dtypes(include=np.number).columns
            df_cleaned[num_cols] = imputer.fit_transform(df_cleaned[num_cols])
            logging.info("Applied KNN imputation to numerical columns")
        elif strategy in ['mean', 'median', 'most_frequent']:
            imputer = SimpleImputer(strategy=strategy)
            num_cols = df_cleaned.select_dtypes(include=np.number).columns
            df_cleaned[num_cols] = imputer.fit_transform(df_cleaned[num_cols])
            logging.info(f"Imputed numerical columns with {strategy}")
        elif strategy == 'constant':
            df_cleaned.fillna(fill_value, inplace=True)
            logging.info(f"Filled missing values with {fill_value}")
        else:
            logging.warning(f"Unsupported strategy: {strategy}")

        return df_cleaned

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates with automatic column selection"""
        original_rows = df.shape[0]
        df_cleaned = df.drop_duplicates()
        removed = original_rows - df_cleaned.shape[0]
        if removed > 0:
            logging.info(f"Removed {removed} duplicate rows")
        return df_cleaned

    def correct_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Automatically detect and correct data types"""
        df_cleaned = df.copy()
        type_changes = {}
        
        for col in df_cleaned.columns:
            # Attempt numeric conversion
            if df_cleaned[col].dtype == object:
                try:
                    # Try converting to numeric
                    converted = pd.to_numeric(df_cleaned[col], errors='coerce')
                    if not converted.isna().all():
                        df_cleaned[col] = converted
                        type_changes[col] = 'numeric'
                        continue
                except:
                    pass
                
                # Attempt datetime conversion
                try:
                    converted = pd.to_datetime(df_cleaned[col], errors='coerce')
                    if not converted.isna().all():
                        df_cleaned[col] = converted
                        type_changes[col] = 'datetime'
                        continue
                except:
                    pass
                
                # Convert to categorical if low cardinality
                unique_ratio = df_cleaned[col].nunique() / len(df_cleaned)
                if unique_ratio < 0.5:  # Less than 50% unique values
                    df_cleaned[col] = df_cleaned[col].astype('category')
                    type_changes[col] = 'category'
        
        if type_changes:
            logging.info(f"Converted data types: {type_changes}")
        return df_cleaned

    def clean_categorical_data(self, df: pd.DataFrame, max_categories: int = 20) -> pd.DataFrame:
        """Clean categorical data by combining rare categories"""
        df_clean = df.copy()
        
        for col in df_clean.select_dtypes(include=['object', 'category']).columns:
            # Skip high-cardinality columns
            if df_clean[col].nunique() > 100:
                continue
                
            # Combine rare categories
            counts = df_clean[col].value_counts()
            rare_categories = counts[counts / len(df_clean) < 0.05].index
            if len(rare_categories) > 0:
                df_clean[col] = df_clean[col].replace(rare_categories, 'Other')
                logging.info(f"Combined {len(rare_categories)} rare categories in '{col}'")
        
        return df_clean

    def clean_text_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean all text columns in the DataFrame"""
        df_clean = df.copy()
        
        for col in df_clean.select_dtypes(include='object').columns:
            df_clean[col] = df_clean[col].astype(str).apply(
                lambda x: re.sub(r'[^\w\s]', '', x)  # Remove special chars
            ).apply(
                lambda x: re.sub(r'\s+', ' ', x)      # Remove extra spaces
            ).str.strip().str.lower()
        
        logging.info("Cleaned text in all string columns")
        return df_clean
    
    def auto_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform all cleaning operations automatically"""
        logging.info("Starting automatic data cleaning")
        df = self.handle_missing_values(df, strategy='auto')
        df = self.remove_duplicates(df)
        df = self.correct_data_types(df)
        df = self.clean_categorical_data(df)
        df = self.clean_text_data(df)
        logging.info("Automatic cleaning complete")
        return df