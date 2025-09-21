import streamlit as st
import logging
import os
import shutil
import pandas as pd
import sqlite3
from datetime import datetime
import time

from utils.config import DATA_DIR, SUPPORTED_FILE_TYPES, PROFILES_DIR, INSIGHTS_DIR, CHROMA_DB_PATH, DATABASE_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SessionManager:
    """
    Manages the Streamlit session state and application-wide resources.
    Ensures consistent state across reruns and provides methods for resetting.
    """
    def __init__(self):
        # Initialize core session state variables if they don't exist
        if "data_loaded" not in st.session_state:
            st.session_state.data_loaded = False
        if "dataframe" not in st.session_state:
            st.session_state.dataframe = None
        if "profile_report_path" not in st.session_state:
            st.session_state.profile_report_path = None
        if "uploaded_file_name" not in st.session_state:
            st.session_state.uploaded_file_name = None
        if "current_dataset_id" not in st.session_state:
            st.session_state.current_dataset_id = None
        if "persistent_plots" not in st.session_state:
            st.session_state.persistent_plots = []
        if "last_plot_count" not in st.session_state:
            st.session_state.last_plot_count = 0
        
        # Initialize managers if they don't exist
        if "chat_memory_manager" not in st.session_state:
            from utils.memory_manager import ChatMemoryManager
            st.session_state.chat_memory_manager = ChatMemoryManager()
        if "llm_handler" not in st.session_state:
            from utils.llm_handler import LLMHandler
            st.session_state.llm_handler = LLMHandler()
        if "rag_pipeline" not in st.session_state:
            from utils.rag_pipeline import RAGPipeline
            st.session_state.rag_pipeline = RAGPipeline(
                df=st.session_state.dataframe,
                chat_memory_manager=st.session_state.chat_memory_manager,
                temperature=st.session_state.llm_handler.temperature,
                top_p=st.session_state.llm_handler.top_p
            )
        
        logging.info("SessionManager initialized.")
        
        # Initialize dataset database
        self._init_dataset_database()

    def _init_dataset_database(self):
        """Initialize dataset database if not exists"""
        if not os.path.exists(DATABASE_PATH):
            conn = sqlite3.connect(DATABASE_PATH)
            c = conn.cursor()
            c.execute('''CREATE TABLE uploaded_files
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          file_name TEXT UNIQUE,
                          upload_time TIMESTAMP,
                          rows INTEGER,
                          columns INTEGER,
                          description TEXT)''')
            conn.commit()
            conn.close()

    def is_data_loaded(self) -> bool:
        return st.session_state.data_loaded

    def set_data_loaded(self, status: bool):
        st.session_state.data_loaded = status
        logging.info(f"Data loaded status set to {status}.")

    def get_dataframe(self) -> pd.DataFrame:
        return st.session_state.dataframe

    def set_dataframe(self, df: pd.DataFrame, file_name: str = None):
        st.session_state.dataframe = df
        self.set_data_loaded(True)
        # Update RAG pipeline with new DataFrame
        st.session_state.rag_pipeline.update_dataframe(df, file_name)
        logging.debug("DataFrame updated in session state.")

    def get_profile_report_path(self) -> str:
        return st.session_state.profile_report_path

    def set_profile_report_path(self, path: str):
        st.session_state.profile_report_path = path
        logging.info(f"Profile report path set to {path}.")

    def record_dataset(self, file_name: str, df: pd.DataFrame, description: str = ""):
        """Record dataset in database and return dataset ID"""
        conn = sqlite3.connect(DATABASE_PATH)
        c = conn.cursor()
        try:
            c.execute('''INSERT INTO uploaded_files 
                         (file_name, upload_time, rows, columns, description)
                         VALUES (?, datetime('now'), ?, ?, ?)''',
                      (file_name, df.shape[0], df.shape[1], description))
            conn.commit()
            dataset_id = c.lastrowid
            logging.info(f"Recorded dataset {file_name} in database with ID {dataset_id}")
            return dataset_id
        except sqlite3.IntegrityError:
            logging.warning(f"Dataset {file_name} already exists in database")
            # Get existing ID
            c.execute("SELECT id FROM uploaded_files WHERE file_name=?", (file_name,))
            result = c.fetchone()
            return result[0] if result else None
        finally:
            conn.close()

    def get_dataset_history(self):
        """Get history of uploaded datasets"""
        conn = sqlite3.connect(DATABASE_PATH)
        c = conn.cursor()
        c.execute("SELECT id, file_name, upload_time, rows, columns, description FROM uploaded_files")
        datasets = c.fetchall()
        conn.close()
        return datasets

    def get_dataset_by_id(self, dataset_id: int):
        """Get dataset metadata by ID"""
        conn = sqlite3.connect(DATABASE_PATH)
        c = conn.cursor()
        c.execute("SELECT id, file_name, upload_time, rows, columns, description FROM uploaded_files WHERE id=?", (dataset_id,))
        dataset = c.fetchone()
        conn.close()
        return dataset

    def reset_session(self):
        logging.info("Resetting session...")
        
        # Reset data-related state
        st.session_state.data_loaded = False
        st.session_state.dataframe = None
        st.session_state.profile_report_path = None
        st.session_state.uploaded_file_name = None
        st.session_state.current_dataset_id = None
        st.session_state.persistent_plots = []
        st.session_state.last_plot_count = 0
        
        # Reset chat history
        if "messages" in st.session_state:
            st.session_state.messages = []

        # Reset memory manager
        from utils.memory_manager import ChatMemoryManager
        st.session_state.chat_memory_manager = ChatMemoryManager()
        
        # Reset RAG pipeline with current parameters
        from utils.rag_pipeline import RAGPipeline
        st.session_state.rag_pipeline = RAGPipeline(
            df=None,
            chat_memory_manager=st.session_state.chat_memory_manager,
            temperature=st.session_state.llm_handler.temperature,
            top_p=st.session_state.llm_handler.top_p
        )

        # Clean up directories
        for directory in [DATA_DIR, PROFILES_DIR, INSIGHTS_DIR, CHROMA_DB_PATH]:
            if os.path.exists(directory):
                try:
                    for item in os.listdir(directory):
                        item_path = os.path.join(directory, item)
                        if os.path.isfile(item_path):
                            os.remove(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                    logging.info(f"Cleaned up contents of directory: {directory}")
                except Exception as e:
                    logging.warning(f"Could not clean up directory {directory}: {e}")
            os.makedirs(directory, exist_ok=True)
            
        logging.info("Session reset complete.")
    
    def get_new_plots(self):
        """Returns only plots generated since the last response"""
        current_count = len(st.session_state.persistent_plots)
        new_plots = st.session_state.persistent_plots[st.session_state.last_plot_count:]
        st.session_state.last_plot_count = current_count
        return new_plots

    def store_plot(self, plot_path: str):
        """Store plot reference in session state"""
        if 'persistent_plots' not in st.session_state:
            st.session_state.persistent_plots = []
            
        # Only store HTML plots and avoid duplicates
        if plot_path.endswith('.html') and plot_path not in st.session_state.persistent_plots:
            st.session_state.persistent_plots.append(plot_path)
            
            # Maintain only last 20 plots
            if len(st.session_state.persistent_plots) > 20:
                st.session_state.persistent_plots.pop(0)