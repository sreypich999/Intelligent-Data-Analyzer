# Intelligent Data Analyzer

A powerful, AI-powered data analysis application built with Streamlit that enables users to upload CSV/Excel files and interact with their data through natural language queries. The application leverages Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) to provide intelligent insights, generate visualizations, and create comprehensive reports.

## 📸 Screenshots

![Intelligent Data Analyzer Interface](https://github.com/sreypich999/Intelligent-Data-Analyzer/blob/main/photo_2025-09-21_23-25-51.jpg)

*Main interface showing the chat-based data analysis with interactive visualizations*

## 🌟 Features

### Core Functionality
- **File Upload Support**: Upload CSV and Excel files for analysis
- **Natural Language Queries**: Chat interface for asking questions about your data in plain English
- **Dataset Management**: Switch between multiple datasets seamlessly
- **Intelligent Analysis**: AI-powered insights and recommendations

### Data Analysis Capabilities
- **Exploratory Data Analysis (EDA)**: Automatic data profiling and statistical summaries
- **Interactive Visualizations**: Generate HTML-based interactive plots using Plotly
- **Correlation Analysis**: Identify relationships between variables
- **Data Quality Assessment**: Detect missing values, outliers, and data inconsistencies
- **Statistical Insights**: Comprehensive statistical analysis and recommendations

### Reporting & Visualization
- **Multi-format Reports**: Generate reports in HTML, PDF, and DOCX formats
- **Interactive Charts**: Scatter plots, histograms, bar charts, heatmaps, and more
- **Data Profile Reports**: Detailed dataset summaries with column types and statistics
- **Downloadable Content**: All reports and visualizations are downloadable

### Advanced Features
- **Dataset Switching**: Natural language commands to switch between datasets
- **Memory Management**: Persistent chat history and context awareness
- **LLM Integration**: Powered by Google's Gemini 2.0 Flash model
- **Vector Database**: ChromaDB for efficient data retrieval and context
- **Fuzzy Matching**: Intelligent dataset name recognition

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Google Gemini API key (for LLM functionality)

### Setup Instructions

1. **Clone the repository** (if applicable) or navigate to the project directory:
   ```bash
   cd chatbot.V1
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   - Copy `.env` file and update the `GOOGLE_API_KEY` with your actual API key
   - Ensure `CHROMA_DB_PATH` points to your desired ChromaDB storage location

4. **Run the application**:
   ```bash
   streamlit run streamlit_app.py
   ```

5. **Access the app**:
   - Open your browser and navigate to `http://localhost:8501`

## 📖 Usage

### Getting Started
1. **Upload Data**: Use the sidebar to upload a CSV or Excel file
2. **Explore Data**: Ask questions like "What columns do I have?" or "Show me the first 10 rows"
3. **Generate Visualizations**: Request charts with commands like "Create a scatter plot of sales vs profit"
4. **Get Insights**: Ask for analysis recommendations or key patterns

### Chat Commands

#### Dataset Management
- `list datasets` - Show all available datasets
- `switch to sales data` - Switch to a specific dataset
- `use customer file` - Load a different dataset

#### Analysis Commands
- `/profile` - Generate comprehensive data profile reports
- `/insights` - Get AI-powered insights and recommendations
- `/help` - Display available commands and examples
- `/reset` - Clear session and start fresh

#### Example Queries
- "What are the key relationships in my data?"
- "Create an interactive histogram of age distribution"
- "Show me correlations between numerical columns"
- "Generate a pie chart of product categories"
- "What data quality issues should I address?"

### Combined Actions
- "Switch to sales data and show top 10 products"
- "Use customer file and create a pie chart of countries"
- "Load inventory data and analyze stock levels"

## ⚙️ Configuration

### Environment Variables
- `GOOGLE_API_KEY`: Your Google Gemini API key (required)
- `CHROMA_DB_PATH`: Path to ChromaDB storage directory

### LLM Settings
- **Temperature**: Controls creativity (0.0 = deterministic, 1.0 = creative)
- **Top P**: Controls response diversity (0.0 = focused, 1.0 = diverse)

### Directory Structure
```
chatbot.V1/
├── data/
│   └── uploaded_files/     # Uploaded datasets
├── reports/
│   ├── profiles/           # Data profile reports
│   ├── insights/           # Generated insights and plots
│   └── interactive_plots/  # HTML visualizations
├── chroma_db/              # Vector database storage
├── utils/                  # Utility modules
│   ├── config.py          # Configuration settings
│   ├── data_loader.py     # Data loading utilities
│   ├── llm_handler.py     # LLM integration
│   ├── rag_pipeline.py    # RAG implementation
│   ├── plot_generator.py  # Visualization tools
│   └── session_manager.py # Session management
├── streamlit_app.py       # Main application
├── requirements.txt       # Python dependencies
└── .env                   # Environment variables
```

## 🏗️ Architecture

### Core Components

#### 1. Streamlit Frontend (`streamlit_app.py`)
- User interface with sidebar for uploads and settings
- Chat interface for natural language queries
- Real-time display of interactive visualizations

#### 2. Session Management (`utils/session_manager.py`)
- Handles dataset loading and switching
- Manages user sessions and data persistence
- Tracks dataset metadata and history

#### 3. Data Processing (`utils/data_loader.py`)
- Loads CSV and Excel files
- Data cleaning and preprocessing
- Integration with pandas for data manipulation

#### 4. LLM Integration (`utils/llm_handler.py`)
- Google Gemini model integration
- Configurable temperature and top-p parameters
- Error handling and response processing

#### 5. RAG Pipeline (`utils/rag_pipeline.py`)
- Retrieval-Augmented Generation implementation
- ChromaDB vector database integration
- Context-aware query processing with tools

#### 6. Visualization Engine (`utils/plot_generator.py`)
- Interactive Plotly chart generation
- HTML export for embedding in chat
- Support for multiple chart types

#### 7. Memory Management (`utils/memory_manager.py`)
- Chat history persistence
- Context window management
- Message formatting and storage

### Data Flow
1. **Upload**: User uploads CSV/Excel file
2. **Processing**: Data is loaded and stored in session
3. **Vectorization**: Data is processed and stored in ChromaDB
4. **Query**: User asks questions via chat interface
5. **Retrieval**: Relevant data context retrieved from vector DB
6. **Generation**: LLM generates response with retrieved context
7. **Visualization**: Interactive plots generated and displayed
8. **Reporting**: Comprehensive reports created in multiple formats

## 📊 Dependencies

### Core Libraries
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Plotly**: Interactive visualizations
- **Seaborn**: Statistical visualization
- **Matplotlib**: Plotting library

### AI/ML Libraries
- **Google Generative AI**: LLM integration
- **LangChain**: LLM framework and tools
- **ChromaDB**: Vector database
- **Scikit-learn**: Machine learning algorithms
- **SciPy**: Scientific computing

### Document Processing
- **python-docx**: Word document generation
- **FPDF**: PDF generation
- **pdfkit**: PDF conversion utilities
- **ydata-profiling**: Automated data profiling

### Utilities
- **python-dotenv**: Environment variable management
- **FuzzyWuzzy**: String matching for dataset names
- **SQLAlchemy**: Database operations

## 🔧 Development

### Running in Development Mode
```bash
streamlit run streamlit_app.py --server.headless true --server.port 8501
```


## 📞 Support

For questions, issues, or feature requests:
- Check the troubleshooting section above
- Review existing issues on GitHub
- Create a new issue with detailed information

---

**Note**: This application requires a valid Google Gemini API key for full functionality. Interactive visualizations are displayed directly in the chat interface, while PDF and DOCX reports are available for download.

