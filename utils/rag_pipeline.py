import pandas as pd
import numpy as np
import logging
import os
import io
import re
from typing import List, Optional
import sqlite3
import time

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field

from utils.llm_handler import LLMHandler
from utils.memory_manager import ChatMemoryManager
from utils.eda_pipeline import EDAPipeline
from utils.config import DEFAULT_TEMPERATURE, DEFAULT_TOP_P, PROFILES_DIR, INSIGHTS_DIR, MAX_ROWS_TO_DISPLAY, REPORT_FORMATS, DATA_DIR
from utils.data_loader import DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Pydantic models for tool inputs
class DataQueryInput(BaseModel):
    query: str = Field(description="The pandas query to execute")
    max_rows: Optional[int] = Field(MAX_ROWS_TO_DISPLAY, description="Maximum rows to display")

class GenerateProfileInput(BaseModel):
    file_name_prefix: str = Field("uploaded_data", description="Prefix for the report filename")

class GeneratePlotInput(BaseModel):
    plot_type: str = Field(description="Type of plot: histogram, scatterplot, boxplot, barplot, heatmap, pairplot, lineplot, violinplot, piechart, qqplot")
    x_col: Optional[str] = Field(None, description="X-axis column name")
    y_col: Optional[str] = Field(None, description="Y-axis column name")
    column: Optional[str] = Field(None, description="Single column name for hist/box plots")
    hue_col: Optional[str] = Field(None, description="Color encoding column")
    columns: Optional[List[str]] = Field(None, description="List of columns for heatmap or pairplot")
    filename_prefix: str = Field("data_insight", description="Prefix for output filename")
    top_n: Optional[int] = Field(None, description="Number of top values to show")
    color: Optional[str] = Field(None, description="Color customization (hex code, name, or comma-separated list)")

class GetInsightsInput(BaseModel):
    max_insights: Optional[int] = Field(5, description="Maximum number of insights to return")

class DatasetRecallInput(BaseModel):
    file_id: Optional[int] = Field(None, description="ID of the file to recall")
    file_name: Optional[str] = Field(None, description="Name of the file to recall")
    description: Optional[str] = Field(None, description="Description of the file to recall")

class RAGPipeline:
    """
    Enhanced RAG pipeline with modern visualization and detailed insights
    """
    def __init__(self, df: pd.DataFrame = None, chat_memory_manager: ChatMemoryManager = None,
                 temperature: float = DEFAULT_TEMPERATURE, top_p: float = DEFAULT_TOP_P):
        
        self.df = df
        self.chat_memory_manager = chat_memory_manager
        self.llm_handler = LLMHandler(temperature=temperature, top_p=top_p)
        self.eda_pipeline = EDAPipeline()
        self.data_loader = DataLoader()
        self.file_history = self.data_loader.get_file_history()
        self.current_dataset_name = None

        self.temperature = temperature
        self.top_p = top_p
        
        self.agent_executor = self._initialize_agent()
        logging.info("RAGPipeline initialized.")
        if self.df is not None:
            logging.info(f"Initial DataFrame shape: {self.df.shape}")
        
    def _initialize_agent(self):
        """Initializes the LangChain agent with enhanced tools."""
        tools = self._initialize_tools()
        llm_model = self.llm_handler.get_model()

        # Enhanced system message with analysis recommendations
        system_message_content = """
        You are an Intelligent Data Analysis Assistant. Your goal is to help users deeply understand their datasets.
        You have access to powerful tools for data exploration, visualization, and insight generation.

        Key Responsibilities:
        1. **Data Context Awareness**: Always consider the current DataFrame context (columns, types, size).
        2. **Tool Utilization**:
            - Use `get_dataframe_info_tool` for dataset overview.
            - Use `data_query` for specific data extraction (limit rows with `max_rows` parameter).
            - Use `generate_profile_report_tool` for comprehensive data profiles (PDF/DOCX).
            - Use `generate_insight_plot` for modern visualizations (Plotly interactive charts).
            - Use `get_data_insights_tool` for key insights and analysis recommendations.
            - Use `recall_dataset` to load previously uploaded datasets by ID, name, or description.
        3. **Dataset Recall**:
            - When user refers to a previous dataset (e.g., "the sales data", "my customer dataset"), 
              use `recall_dataset` to load it.
            - After switching, confirm the dataset change and provide a summary of the new dataset.
        4. **Insight Generation**:
            - Always provide detailed interpretations of results.
            - Highlight relationships between columns.
            - Suggest next steps for analysis (e.g., "This strong correlation suggests...")
        5. **Visualization Best Practices**:
            - Prefer interactive Plotly charts for richer exploration.
            - When users request top N values (e.g., "top 10", "top 5"), use the `top_n` parameter
            - Apply this to bar plots, pie charts, and other categorical visualizations
            - Example: "Show top 10 products by sales" -> use top_n=10 with bar plot
        6. **Analysis Recommendations**:
            - Suggest machine learning approaches when appropriate.
            - Recommend data cleaning steps for missing values.
            - Propose feature engineering ideas.
        7. **User Guidance**:
            - Explain complex concepts in simple terms.
            - Provide context for your recommendations.
            - Always offer to create visualizations to support your insights.

        Important: After generating any output, explain its significance and suggest next steps.
        After switching datasets, always provide a summary of the new dataset and ask if the user wants to proceed with analysis.
        """

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_message_content),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(llm_model, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
        logging.info("LangChain agent initialized.")
        return agent_executor

    def _initialize_tools(self):
        """Defines the enhanced tools available to the LLM agent."""
        
        @tool
        def get_dataframe_info_tool() -> dict:
            """Provides a comprehensive summary of the loaded DataFrame."""
            if self.df is None:
                return {"error": "No DataFrame is currently loaded. Please upload a file first."}
            return self.eda_pipeline.get_dataframe_summary(self.df)

        @tool(args_schema=DataQueryInput)
        def data_query(query: str, max_rows: int = MAX_ROWS_TO_DISPLAY) -> str:
            """
            Executes a pandas operation on the loaded DataFrame with row limit.
            Examples: "df.head()", "df.groupby('Category')['Value'].mean()"
            """
            if self.df is None:
                return "Error: No DataFrame is loaded. Please upload a file first."
            
            try:
                # Limit the output to max_rows
                query = self._add_row_limit(query, max_rows)
                
                # Capture stdout to return print statements
                old_stdout = io.StringIO()
                import sys
                sys.stdout = old_stdout

                result = eval(query, {'pd': pd, 'np': np, 'df': self.df.copy()})
                
                sys.stdout = sys.__stdout__
                printed_output = old_stdout.getvalue()

                # Format output
                if isinstance(result, pd.DataFrame):
                    return f"Query Result (First {min(len(result), max_rows)} rows):\n{result.head(max_rows).to_string()}"
                elif isinstance(result, pd.Series):
                    return f"Query Result (First {min(len(result), max_rows)} items):\n{result.head(max_rows).to_string()}"
                else:
                    return f"Query Result: {printed_output if printed_output else str(result)}"
            except Exception as e:
                return f"Error executing query: {e}"

        @tool(args_schema=GenerateProfileInput)
        def generate_profile_report_tool(file_name_prefix: str = "uploaded_data") -> str:
            """Generates comprehensive data profile reports in DOCX and PDF formats."""
            if self.df is None:
                return "Error: No DataFrame is loaded. Please upload a file first."
            try:
                report_paths = self.eda_pipeline.generate_profile_report(self.df, file_name_prefix)
                if report_paths:
                    paths_str = "\n".join(
                        [f"- {fmt.upper()}: [REPORT_{fmt.upper()}_PATH:{path}]" 
                         for fmt in REPORT_FORMATS if fmt in report_paths]
                    )
                    return f"Profile reports generated:\n{paths_str}"
                return "Failed to generate profile reports."
            except Exception as e:
                return f"Error generating reports: {e}"

        @tool(args_schema=GeneratePlotInput)
        def generate_insight_plot(
            plot_type: str, 
            x_col: Optional[str] = None, 
            y_col: Optional[str] = None, 
            column: Optional[str] = None, 
            hue_col: Optional[str] = None, 
            columns: Optional[List[str]] = None, 
            filename_prefix: str = "data_insight",
            top_n: Optional[int] = None,
            color: Optional[str] = None
        ) -> str:
            """Generates modern interactive visualizations using Plotly with color customization."""
            if self.df is None:
                return "Error: No DataFrame is loaded. Please upload a file first."
                
            logging.info(f"Generating {plot_type} plot for dataset: {self.current_dataset_name}")

            column_args = {
                'x_col': x_col,
                'y_col': y_col,
                'column': column,
                'hue_col': hue_col,
                'columns': columns,
                'top_n': top_n
            }

            try:
                plot_paths = self.eda_pipeline.generate_insight_plot(
                    self.df, plot_type, column_args, file_name_prefix=filename_prefix, color=color
                )
                
                if plot_paths:
                    paths_str = "\n".join([f"- {fmt.upper()}: [PLOT_{fmt.upper()}_PATH:{path}]" for fmt, path in plot_paths.items()])
                    return f"I have generated a {plot_type} plot:\n{paths_str}"
                return f"Failed to generate {plot_type} plot. Check column names/types."
            except Exception as e:
                return f"Error generating plot: {e}"

        @tool(args_schema=GetInsightsInput)
        def get_data_insights_tool(max_insights: int = 5) -> str:
            """Provides key insights and analysis recommendations."""
            if self.df is None:
                return "Error: No DataFrame is loaded. Please upload a file first."
            
            try:
                insights = self.eda_pipeline.get_insight_summary(self.df)
                relationships = self.eda_pipeline.detect_relationships(self.df)[:max_insights]
                recommendations = self.eda_pipeline.generate_recommendations(self.df)[:max_insights]
                
                response = "## Data Insights and Recommendations\n\n"
                response += insights + "\n\n"
                
                if relationships:
                    response += "### Key Relationships Found:\n"
                    for rel in relationships:
                        response += f"- {rel['description']} (Strength: {rel.get('strength', rel.get('chi2', 'N/A'))})\n"
                        response += f"  Suggestion: {rel['suggestion']}\n"
                    response += "\n"
                
                if recommendations:
                    response += "### Analysis Recommendations:\n"
                    for rec in recommendations:
                        response += f"- **{rec['type'].replace('_', ' ').title()}** ({rec['priority']} priority): {rec['description']}\n"
                        response += f"  Suggestion: {rec['suggestion']}\n"
                
                return response
            except Exception as e:
                return f"Error generating insights: {e}"

        @tool(args_schema=DatasetRecallInput)
        def recall_dataset(file_id: int = None, file_name: str = None, description: str = None) -> str:
            """Recalls a previously uploaded dataset by ID, name, or description"""
            try:
                # Get file metadata
                conn = sqlite3.connect('file_database.db')
                c = conn.cursor()
                
                # Build query based on provided parameters
                if file_id:
                    c.execute("SELECT id, file_name FROM uploaded_files WHERE id=?", (file_id,))
                elif file_name:
                    c.execute("SELECT id, file_name FROM uploaded_files WHERE file_name LIKE ?", (f"%{file_name}%",))
                elif description:
                    c.execute("SELECT id, file_name FROM uploaded_files WHERE description LIKE ?", (f"%{description}%",))
                else:
                    # If no parameters, return list of datasets
                    c.execute("SELECT id, file_name, rows, columns, description FROM uploaded_files")
                    datasets = c.fetchall()
                    conn.close()
                    if datasets:
                        response = "Available datasets:\n"
                        for did, name, rows, cols, desc in datasets:
                            response += f"- ID {did}: {name} ({rows}x{cols}) - {desc or 'No description'}\n"
                        return response
                    return "No datasets available in database"
                
                result = c.fetchone()
                conn.close()
                
                if not result:
                    return "No matching dataset found"
                    
                file_id, file_name = result
                file_path = os.path.join(DATA_DIR, file_name)
                
                # Load file
                if not os.path.exists(file_path):
                    return f"File not found: {file_name}"
                    
                if file_name.endswith('.csv'):
                    df = self.data_loader.load_csv(file_path)
                elif file_name.endswith('.xlsx'):
                    df = self.data_loader.load_excel(file_path)
                else:
                    return "Unsupported file type"
                
                if df is not None:
                    self.df = df
                    self.current_dataset_name = file_name
                    return f"Successfully loaded dataset: {file_name} with {df.shape[0]} rows and {df.shape[1]} columns"
                return "Failed to load dataset"
            except Exception as e:
                return f"Error recalling dataset: {e}"

        return [
            get_dataframe_info_tool, 
            data_query, 
            generate_profile_report_tool, 
            generate_insight_plot,
            get_data_insights_tool,
            recall_dataset
        ]

    def _add_row_limit(self, query: str, max_rows: int) -> str:
        """Adds row limitation to DataFrame display queries."""
        if "head(" in query or "tail(" in query or "sample(" in query:
            return query
            
        if "df[" in query and "]" in query:
            return f"{query}.head({max_rows})"
            
        return query

    def update_dataframe(self, df: pd.DataFrame, dataset_name: str = None):
        self.df = df
        self.current_dataset_name = dataset_name
        # Reinitialize tools with new dataframe
        self.agent_executor = self._initialize_agent() 
        logging.info(f"RAGPipeline: DataFrame updated. New shape: {self.df.shape if self.df is not None else 'None'}")

    def process_query(self, user_query: str) -> str:
        """Processes user queries with enhanced insight generation."""
        if self.df is None and not user_query.lower().strip() in ["what can you do?", "help", "commands"]:
            return "Please upload a dataset first or switch to an existing dataset. Type 'help' for assistance."

        chat_history = self.chat_memory_manager.get_langchain_messages()

        try:
            response = self.agent_executor.invoke({
                "input": user_query,
                "chat_history": chat_history
            })
            return response.get('output', 'No response from agent.')
        except Exception as e:
            logging.error(f"Error processing query: {e}")
            return f"I'm sorry, I encountered an error: {e}"