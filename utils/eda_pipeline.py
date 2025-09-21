import pandas as pd
import numpy as np
import logging
import os
import json
import base64
from datetime import datetime
from ydata_profiling import ProfileReport
from utils.plot_generator import PlotGenerator
from utils.config import PROFILES_DIR, INSIGHTS_DIR, MAX_ROWS_TO_DISPLAY, REPORT_FORMATS
from scipy.stats import chi2_contingency, f_oneway
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from docx import Document
from docx.shared import Inches
import pdfkit
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EDAPipeline:
    """
    Enhanced EDA pipeline with more graphs and insights
    """
    def __init__(self):
        self.plot_generator = PlotGenerator()
        logging.info("EDAPipeline initialized.")

    def generate_profile_report(self, df: pd.DataFrame, file_name_prefix: str = "uploaded_data") -> dict:
        """
        Generates comprehensive data profile reports in DOCX and PDF formats.
        :param df: The DataFrame to profile.
        :param file_name_prefix: Prefix for the report filename.
        :return: Dictionary with paths to generated reports.
        """
        if df is None or df.empty:
            logging.warning("Cannot generate profile report for empty DataFrame.")
            return {}
        
        # Generate HTML profile report
        profile = ProfileReport(df, title=f"Profiling Report for {file_name_prefix}", explorative=True)
        html_report = profile.to_html()
        
        report_paths = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save HTML report
        html_path = os.path.join(PROFILES_DIR, f"{file_name_prefix}_{timestamp}.html")
        profile.to_file(html_path)
        report_paths['html'] = html_path
        logging.info(f"HTML profile report saved to: {html_path}")
        
        # Convert to DOCX
        if 'docx' in REPORT_FORMATS:
            try:
                from utils.report_formatter import html_to_docx
                docx_path = os.path.join(PROFILES_DIR, f"{file_name_prefix}_{timestamp}.docx")
                html_to_docx(html_path, docx_path)
                report_paths['docx'] = docx_path
                logging.info(f"DOCX profile report saved to: {docx_path}")
            except Exception as e:
                logging.error(f"Error generating DOCX report: {e}")
        
        # Convert to PDF
        if 'pdf' in REPORT_FORMATS:
            try:
                pdf_path = os.path.join(PROFILES_DIR, f"{file_name_prefix}_{timestamp}.pdf")
                options = {
                    'page-size': 'A4',
                    'margin-top': '0.75in',
                    'margin-right': '0.75in',
                    'margin-bottom': '0.75in',
                    'margin-left': '0.75in',
                    'encoding': "UTF-8",
                }
                pdfkit.from_file(html_path, pdf_path, options=options)
                report_paths['pdf'] = pdf_path
                logging.info(f"PDF profile report saved to: {pdf_path}")
            except Exception as e:
                logging.error(f"Error generating PDF report: {e}")
        
        return report_paths

    def generate_insight_plot(self, df: pd.DataFrame, plot_type: str, column_args: dict, 
                             file_name_prefix: str = "data_insight", color: str = None) -> dict:
        """
        Generates a specific plot-based data insight with top N filtering and color customization.
        Returns a dictionary of plot paths.
        """
        if df is None or df.empty:
            logging.warning(f"DataFrame is empty or None. Cannot generate '{plot_type}' insight.")
            return None

        safe_prefix = "".join([c for c in file_name_prefix if c.isalnum() or c in (' ', '-', '_')]).replace(' ', '_')
        
        try:
            plot_paths = None
            if plot_type == 'histogram':
                column = column_args.get('column')
                plot_paths = self.plot_generator.generate_histogram(df, column, safe_prefix, color)
            elif plot_type == 'scatterplot':
                x_col = column_args.get('x_col')
                y_col = column_args.get('y_col')
                hue_col = column_args.get('hue_col')
                plot_paths = self.plot_generator.generate_scatterplot(df, x_col, y_col, hue_col, safe_prefix, color)
            elif plot_type == 'boxplot':
                column = column_args.get('column')
                by_col = column_args.get('by_col')
                plot_paths = self.plot_generator.generate_boxplot(df, column, by_col, safe_prefix, color)
            elif plot_type == 'barplot':
                x_col = column_args.get('x_col')
                y_col = column_args.get('y_col')
                top_n = column_args.get('top_n')
                plot_paths = self.plot_generator.generate_barplot(df, x_col, y_col, top_n, safe_prefix, color)
            elif plot_type == 'heatmap':
                columns_for_heatmap = column_args.get('columns')
                plot_paths = self.plot_generator.generate_heatmap(df, columns_for_heatmap, safe_prefix)
            elif plot_type == 'pairplot':
                columns = column_args.get('columns')
                hue_col = column_args.get('hue_col')
                plot_paths = self.plot_generator.generate_pairplot(df, columns, hue_col, safe_prefix, color)
            elif plot_type == 'lineplot':
                x_col = column_args.get('x_col')
                y_col = column_args.get('y_col')
                hue_col = column_args.get('hue_col')
                plot_paths = self.plot_generator.generate_lineplot(df, x_col, y_col, hue_col, safe_prefix, color)
            elif plot_type == 'violinplot':
                column = column_args.get('column')
                by_col = column_args.get('by_col')
                plot_paths = self.plot_generator.generate_violinplot(df, column, by_col, safe_prefix, color)
            elif plot_type == 'piechart':
                column = column_args.get('column')
                top_n = column_args.get('top_n')
                plot_paths = self.plot_generator.generate_piechart(df, column, top_n, safe_prefix, color)
            elif plot_type == 'qqplot':
                column = column_args.get('column')
                plot_paths = self.plot_generator.generate_qqplot(df, column, safe_prefix)
            else:
                logging.warning(f"Unknown plot type requested: {plot_type}")
                return None
            
            return plot_paths

        except Exception as e:
            logging.error(f"Error generating {plot_type} plot insight: {e}", exc_info=True)
            return None

    def generate_all_plots(self, df: pd.DataFrame, prefix: str = "full_analysis"):
        """Generates a comprehensive set of plots for all columns"""
        plot_paths = {}
        
        # Numerical columns
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        for col in num_cols:
            try:
                # Histogram
                plot_paths[f'hist_{col}'] = self.plot_generator.generate_histogram(df, col, f"{prefix}_hist_{col}")
                
                # Box plot
                plot_paths[f'box_{col}'] = self.plot_generator.generate_boxplot(df, col, None, f"{prefix}_box_{col}")
                
                # QQ plot
                plot_paths[f'qq_{col}'] = self.plot_generator.generate_qqplot(df, col, f"{prefix}_qq_{col}")
            except Exception as e:
                logging.error(f"Error generating plots for {col}: {e}")
        
        # Categorical columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in cat_cols:
            try:
                # Bar plot
                plot_paths[f'bar_{col}'] = self.plot_generator.generate_barplot(df, col, None, 10, f"{prefix}_bar_{col}")
                
                # Pie chart
                if df[col].nunique() <= 20:  # Only for reasonable number of categories
                    plot_paths[f'pie_{col}'] = self.plot_generator.generate_piechart(df, col, 10, f"{prefix}_pie_{col}")
            except Exception as e:
                logging.error(f"Error generating plots for {col}: {e}")
        
        # Pairwise relationships
        if len(num_cols) >= 2:
            try:
                # Correlation heatmap
                plot_paths['heatmap'] = self.plot_generator.generate_heatmap(df, num_cols, f"{prefix}_heatmap")
                
                # Scatter matrix
                if len(num_cols) <= 8:  # Limit to avoid too many plots
                    plot_paths['pairplot'] = self.plot_generator.generate_pairplot(df, num_cols[:8], None, f"{prefix}_pairplot")
            except Exception as e:
                logging.error(f"Error generating relationship plots: {e}")
        
        # Time series plots if datetime columns exist
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_cols and len(num_cols) > 0:
            for date_col in date_cols:
                for num_col in num_cols[:3]:  # Limit to first 3 numerical columns
                    try:
                        plot_paths[f'line_{date_col}_{num_col}'] = self.plot_generator.generate_lineplot(
                            df, date_col, num_col, None, f"{prefix}_line_{date_col}_{num_col}"
                        )
                    except Exception as e:
                        logging.error(f"Error generating time series plot for {date_col} and {num_col}: {e}")
        
        logging.info(f"Generated {len(plot_paths)} plots for comprehensive analysis")
        return plot_paths

    def get_dataframe_summary(self, df: pd.DataFrame) -> dict:
        """Provides a comprehensive summary of the DataFrame."""
        if df is None or df.empty:
            return {"error": "No DataFrame loaded."}
        
        summary = {
            "shape": f"{df.shape[0]} rows, {df.shape[1]} columns",
            "columns": [],
            "missing_values": {},
            "data_types": {},
            "sample_data": {}
        }
        
        # Column information
        for col in df.columns:
            dtype = str(df[col].dtype)
            unique_count = df[col].nunique()
            missing_count = df[col].isnull().sum()
            missing_percent = (missing_count / len(df)) * 100
            
            summary["columns"].append(col)
            summary["data_types"][col] = dtype
            summary["missing_values"][col] = {
                "count": missing_count,
                "percent": round(missing_percent, 2)
            }
            
            # Sample data (limit to MAX_ROWS_TO_DISPLAY)
            if dtype in ['object', 'category', 'bool']:
                sample = df[col].value_counts().head(10).to_dict()
            else:
                sample = {
                    "min": round(df[col].min(), 4),
                    "max": round(df[col].max(), 4),
                    "mean": round(df[col].mean(), 4),
                    "median": round(df[col].median(), 4)
                }
            summary["sample_data"][col] = sample
        
        return summary

    def detect_relationships(self, df: pd.DataFrame) -> list:
        """Detects potential relationships between columns."""
        if df is None or df.empty:
            return []
        
        relationships = []
        numerical_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
        
        # Numerical vs Numerical: Correlation
        if len(numerical_cols) > 1:
            corr_matrix = df[numerical_cols].corr().abs().stack().reset_index()
            corr_matrix.columns = ['var1', 'var2', 'correlation']
            corr_matrix = corr_matrix[corr_matrix['var1'] != corr_matrix['var2']]
            strong_corr = corr_matrix[corr_matrix['correlation'] > 0.7].sort_values('correlation', ascending=False)
            
            for _, row in strong_corr.iterrows():
                relationships.append({
                    "type": "numerical-numerical",
                    "columns": [row['var1'], row['var2']],
                    "strength": round(row['correlation'], 2),
                    "description": f"Strong positive correlation between {row['var1']} and {row['var2']}",
                    "suggestion": "Consider creating a scatter plot to visualize this relationship"
                })
        
        # Categorical vs Numerical: ANOVA
        for cat_col in categorical_cols:
            for num_col in numerical_cols:
                groups = [group for group in df.groupby(cat_col)[num_col] if group[1].count() > 1]
                if len(groups) > 1:
                    group_data = [group[1] for group in groups]
                    f_stat, p_val = f_oneway(*group_data)
                    if p_val < 0.05:
                        relationships.append({
                            "type": "categorical-numerical",
                            "columns": [cat_col, num_col],
                            "strength": f_stat,
                            "p_value": p_val,
                            "description": f"Significant difference in {num_col} across groups of {cat_col}",
                            "suggestion": "Consider creating a box plot or violin plot to visualize the differences"
                        })
        
        # Categorical vs Categorical: Chi-square
        if len(categorical_cols) > 1:
            for i, col1 in enumerate(categorical_cols):
                for col2 in categorical_cols[i+1:]:
                    contingency = pd.crosstab(df[col1], df[col2])
                    if contingency.size > 0:
                        chi2, p_val, _, _ = chi2_contingency(contingency)
                        if p_val < 0.05:
                            relationships.append({
                                "type": "categorical-categorical",
                                "columns": [col1, col2],
                                "chi2": chi2,
                                "p_value": p_val,
                                "description": f"Significant association between {col1} and {col2}",
                                "suggestion": "Consider creating a stacked bar chart or mosaic plot to visualize this relationship"
                            })
        
        return relationships

    def generate_recommendations(self, df: pd.DataFrame) -> list:
        """Generates data analysis recommendations based on the dataset."""
        if df is None or df.empty:
            return []
        
        recommendations = []
        numerical_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
        
        # Data cleaning recommendations
        missing_cols = [col for col in df.columns if df[col].isnull().sum() > 0]
        if missing_cols:
            recommendations.append({
                "type": "data_cleaning",
                "priority": "high",
                "description": f"Missing values found in {len(missing_cols)} columns",
                "suggestion": "Consider imputation techniques (mean, median, mode) or removal of missing values"
            })
        
        # Outlier detection
        if len(numerical_cols) > 0:
            recommendations.append({
                "type": "analysis",
                "priority": "medium",
                "description": "Numerical columns present in dataset",
                "suggestion": "Perform outlier detection using box plots or z-scores"
            })
        
        # Feature engineering
        if len(df.columns) > 5:
            recommendations.append({
                "type": "feature_engineering",
                "priority": "medium",
                "description": "Dataset has multiple features",
                "suggestion": "Consider creating interaction terms or polynomial features for machine learning"
            })
        
        # Machine learning recommendations
        if len(categorical_cols) > 0 and len(numerical_cols) > 0:
            target_candidates = []
            if len(categorical_cols) > 0:
                target_candidates.append(f"classification on '{categorical_cols[0]}'")
            if len(numerical_cols) > 0:
                target_candidates.append(f"regression on '{numerical_cols[0]}'")
            
            if target_candidates:
                recommendations.append({
                    "type": "machine_learning",
                    "priority": "high",
                    "description": "Dataset has both numerical and categorical features",
                    "suggestion": f"Consider building a predictive model for {', '.join(target_candidates)} using algorithms like Random Forest or Gradient Boosting"
                })
        
        # Time series analysis
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_cols:
            recommendations.append({
                "type": "time_series",
                "priority": "high",
                "description": f"Date/time columns detected: {', '.join(date_cols)}",
                "suggestion": "Perform time series analysis to identify trends, seasonality, and patterns over time"
            })
        
        # Dimensionality reduction
        if len(df.columns) > 10:
            recommendations.append({
                "type": "dimensionality_reduction",
                "priority": "medium",
                "description": "Dataset has a high number of features",
                "suggestion": "Apply dimensionality reduction techniques like PCA or t-SNE to visualize high-dimensional data"
            })
        
        return recommendations

    def get_insight_summary(self, df: pd.DataFrame) -> str:
        """Provides a textual summary of key insights about the DataFrame."""
        if df is None or df.empty:
            return "No DataFrame loaded."
        
        summary = "## Dataset Insights Summary\n\n"
        summary += f"- **Shape**: {df.shape[0]} rows, {df.shape[1]} columns\n"
        
        # Data types summary
        dtype_counts = df.dtypes.value_counts().to_dict()
        summary += "- **Data Types**:\n"
        for dtype, count in dtype_counts.items():
            summary += f"  - {dtype}: {count} columns\n"
        
        # Missing values summary
        missing_cols = [col for col in df.columns if df[col].isnull().sum() > 0]
        if missing_cols:
            summary += f"- **Missing Values**: {len(missing_cols)} columns have missing values\n"
        else:
            summary += "- **Missing Values**: No missing values found\n"
        
        # Relationships summary
        relationships = self.detect_relationships(df)
        if relationships:
            summary += f"- **Key Relationships**: {len(relationships)} significant relationships detected\n"
        else:
            summary += "- **Key Relationships**: No strong relationships detected\n"
        
        # Recommendations summary
        recommendations = self.generate_recommendations(df)
        if recommendations:
            summary += f"- **Recommendations**: {len(recommendations)} actionable suggestions for further analysis\n"
        
        return summary