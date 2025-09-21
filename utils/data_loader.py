import pandas as pd
import logging
import os
import sqlite3
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    """Enhanced data loader with database tracking"""
    def __init__(self):
        self.db_path = self._get_db_path()
        self._init_database()
        logging.info("DataLoader initialized with database tracking")

    def _get_db_path(self):
        """Get path to database file"""
        from utils.config import DATABASE_PATH
        return DATABASE_PATH

    def _get_connection(self):
        """Create a new database connection for each operation"""
        return sqlite3.connect(self.db_path)

    def _init_database(self):
        """Initialize the SQLite database for file tracking"""
        conn = self._get_connection()
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS uploaded_files
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      file_name TEXT UNIQUE,
                      upload_time TIMESTAMP,
                      rows INTEGER,
                      columns INTEGER,
                      description TEXT)''')
        conn.commit()
        conn.close()

    def record_file_metadata(self, file_name: str, df: pd.DataFrame, description: str = ""):
        """Record file metadata in the database"""
        conn = self._get_connection()
        c = conn.cursor()
        try:
            c.execute('''INSERT INTO uploaded_files 
                         (file_name, upload_time, rows, columns, description)
                         VALUES (?, ?, ?, ?, ?)''',
                      (file_name, datetime.now(), df.shape[0], df.shape[1], description))
            conn.commit()
            logging.info(f"Recorded metadata for {file_name} in database")
            return c.lastrowid
        except sqlite3.IntegrityError:
            logging.warning(f"File {file_name} already exists in database")
            c.execute("SELECT id FROM uploaded_files WHERE file_name=?", (file_name,))
            result = c.fetchone()
            return result[0] if result else None
        except Exception as e:
            logging.error(f"Error recording file metadata: {e}")
            return None
        finally:
            conn.close()

    def get_file_history(self):
        """Get history of uploaded files"""
        conn = self._get_connection()
        c = conn.cursor()
        c.execute("SELECT id, file_name, upload_time, rows, columns, description FROM uploaded_files")
        result = c.fetchall()
        conn.close()
        return result

    def get_file_by_id(self, file_id: int):
        """Get file metadata by ID"""
        conn = self._get_connection()
        c = conn.cursor()
        c.execute("SELECT id, file_name, upload_time, rows, columns, description FROM uploaded_files WHERE id=?", (file_id,))
        result = c.fetchone()
        conn.close()
        return result

    def load_csv(self, file_path: str) -> pd.DataFrame:
        """
        Loads a CSV file into a pandas DataFrame.
        Attempts to infer encoding if standard UTF-8 fails.
        """
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {os.path.basename(file_path)}")
        try:
            df = pd.read_csv(file_path)
            logging.info(f"CSV loaded successfully from {file_path}. Shape: {df.shape}")
            return df
        except UnicodeDecodeError:
            logging.warning(f"UnicodeDecodeError with {file_path}. Attempting to infer encoding.")
            try:
                # Try common encodings
                df = pd.read_csv(file_path, encoding='latin1')
                logging.info(f"CSV loaded with latin1 encoding from {file_path}. Shape: {df.shape}")
                return df
            except Exception as e:
                logging.error(f"Failed to load CSV {file_path} with inferred encoding: {e}", exc_info=True)
                raise ValueError(f"Could not decode CSV file {os.path.basename(file_path)}. Try 'latin1' or another encoding.") from e
        except pd.errors.EmptyDataError as e:
            logging.error(f"EmptyDataError: No columns to parse from file {file_path}: {e}")
            raise pd.errors.EmptyDataError(f"The CSV file {os.path.basename(file_path)} is empty or contains no data.") from e
        except Exception as e:
            logging.error(f"Error loading CSV file {file_path}: {e}", exc_info=True)
            raise ValueError(f"Error loading CSV file {os.path.basename(file_path)}: {e}") from e

    def load_excel(self, file_path: str, sheet_name=0) -> pd.DataFrame:
        """
        Loads an Excel file (xlsx, xls) into a pandas DataFrame.
        Defaults to the first sheet.
        """
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {os.path.basename(file_path)}")
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            logging.info(f"Excel loaded successfully from {file_path} (sheet: {sheet_name}). Shape: {df.shape}")
            return df
        except pd.errors.EmptyDataError as e:
             logging.error(f"EmptyDataError: No columns to parse from file {file_path}: {e}")
             raise pd.errors.EmptyDataError(f"The Excel file {os.path.basename(file_path)} is empty or contains no data on sheet '{sheet_name}'.") from e
        except Exception as e:
            logging.error(f"Error loading Excel file {file_path}: {e}", exc_info=True)
            raise ValueError(f"Error loading Excel file {os.path.basename(file_path)}: {e}") from e