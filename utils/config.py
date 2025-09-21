import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data storage directories
DATA_DIR = os.path.join(BASE_DIR, "data", "uploaded_files")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
PROFILES_DIR = os.path.join(REPORTS_DIR, "profiles")
INSIGHTS_DIR = os.path.join(REPORTS_DIR, "insights")
INTERACTIVE_PLOTS_DIR = os.path.join(REPORTS_DIR, "interactive_plots")
PLOT_CACHE_DIR = os.path.join(REPORTS_DIR, "plot_cache")

# Ensure these directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(PROFILES_DIR, exist_ok=True)
os.makedirs(INSIGHTS_DIR, exist_ok=True)
os.makedirs(INTERACTIVE_PLOTS_DIR, exist_ok=True)
os.makedirs(PLOT_CACHE_DIR, exist_ok=True)

# Supported file types for upload
SUPPORTED_FILE_TYPES = ["csv", "xlsx"]

# ChromaDB path
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db")
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

# Database for storing file metadata
DATABASE_PATH = os.path.join(BASE_DIR, "file_database.db")

# LLM Model configuration
GEMINI_MODEL_NAME = "gemini-2.0-flash"
MAX_CHAT_HISTORY_LENGTH = 20
MAX_PLOTS_PER_QUESTION = 5

# Default LLM parameters
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 0.9

# Plotting configuration
MAX_ROWS_TO_DISPLAY = 30
PLOTLY_THEME = "plotly_white"

# Report formats
REPORT_FORMATS = ["html","docx", "pdf"]

# Column name mapping for strange column names
COLUMN_NAME_MAPPING = {
    "strange_col1": "customer_id",
    "weird_col2": "order_date",
}