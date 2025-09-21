import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from scipy import stats
from utils.config import INSIGHTS_DIR, INTERACTIVE_PLOTS_DIR, MAX_ROWS_TO_DISPLAY, PLOTLY_THEME
import colorsys
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PlotGenerator:
    """
    Enhanced plot generator with color customization and data validation
    """
    def __init__(self):
        self.insights_dir = INSIGHTS_DIR
        self.interactive_dir = INTERACTIVE_PLOTS_DIR
        os.makedirs(self.insights_dir, exist_ok=True)
        os.makedirs(self.interactive_dir, exist_ok=True)
        self.color_sequence = px.colors.qualitative.Plotly
        logging.info(f"PlotGenerator initialized. Output directories: {self.insights_dir}, {self.interactive_dir}")

    def _validate_data(self, df: pd.DataFrame) -> bool:
        """Validate dataframe before plotting"""
        if df is None or df.empty:
            logging.error("Cannot generate plot: DataFrame is empty or None")
            return False
        if len(df) < 2:
            logging.error("DataFrame has less than 2 rows, insufficient for visualization")
            return False
        return True

    def _validate_column(self, df: pd.DataFrame, column: str) -> bool:
        """Validate column exists and has data"""
        if column not in df.columns:
            logging.error(f"Column '{column}' not found in DataFrame")
            return False
        if df[column].isnull().all():
            logging.error(f"Column '{column}' contains only null values")
            return False
        return True

    def _save_plot(self, fig, filename: str, plot_type: str = "static") -> dict:
        """Saves plots in multiple formats and returns paths."""
        base_name = os.path.splitext(filename)[0]
        paths = {}
        
        try:
            # Static plot (PNG)
            png_path = os.path.join(self.insights_dir, f"{base_name}.png")
            if hasattr(fig, "write_image"):
                fig.write_image(png_path, engine="kaleido")
            else:
                fig.savefig(png_path, bbox_inches='tight')
                plt.close(fig)
            paths['png'] = png_path
            
            # Interactive plot (HTML) for Plotly figures
            if plot_type == "interactive" and hasattr(fig, "write_html"):
                html_path = os.path.join(self.interactive_dir, f"{base_name}.html")
                fig.write_html(html_path, include_plotlyjs='cdn')
                paths['html'] = html_path
                
            logging.info(f"Plot saved: {png_path}")
            return paths
        except Exception as e:
            logging.error(f"Error saving plot: {e}")
            return None

    def _apply_color_customization(self, fig, color: str = None, color_sequence: list = None):
        """Applies color customization to plotly figure"""
        if color:
            try:
                if color.startswith('#'):
                    # Single color
                    fig.update_traces(marker=dict(color=color), 
                                      selector=dict(type='scatter'))
                    fig.update_traces(marker=dict(color=color), 
                                      selector=dict(type='bar'))
                    fig.update_traces(fillcolor=color, 
                                      selector=dict(type='histogram'))
                elif ',' in color:
                    # Color sequence
                    colors = [c.strip() for c in color.split(',')]
                    fig.update_traces(marker=dict(colors=colors), 
                                      selector=dict(type='pie'))
                    fig.update_layout(colorway=colors)
            except Exception as e:
                logging.error(f"Error applying color customization: {e}")
        return fig

    def generate_histogram(self, df: pd.DataFrame, column: str, filename_prefix: str = "histogram", 
                          color: str = None) -> dict:
        """Generates a histogram for a numerical column."""
        if not self._validate_data(df) or not self._validate_column(df, column) or not pd.api.types.is_numeric_dtype(df[column]):
            return None
        
        # Plotly version
        fig = px.histogram(
            df, 
            x=column, 
            nbins=50, 
            title=f'Distribution of {column}',
            marginal='box',
            color_discrete_sequence=[self.color_sequence[0]],
            template=PLOTLY_THEME
        )
        fig.update_layout(
            xaxis_title=column,
            yaxis_title='Count',
            hovermode='x'
        )
        fig = self._apply_color_customization(fig, color)
        return self._save_plot(fig, f"{filename_prefix}_{column}.html", "interactive")

    def generate_scatterplot(self, df: pd.DataFrame, x_col: str, y_col: str, hue_col: str = None, 
                           filename_prefix: str = "scatter", color: str = None) -> dict:
        """Generates a scatter plot for two numerical columns."""
        if not self._validate_data(df) or not self._validate_column(df, x_col) or not self._validate_column(df, y_col):
            return None
        
        # Plotly version
        fig = px.scatter(
            df, 
            x=x_col, 
            y=y_col, 
            color=hue_col,
            title=f'{y_col} vs {x_col}',
            trendline='ols',
            trendline_color_override='red',
            color_continuous_scale=px.colors.sequential.Viridis,
            template=PLOTLY_THEME
        )
        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_col,
            hovermode='closest'
        )
        fig = self._apply_color_customization(fig, color)
        return self._save_plot(fig, f"{filename_prefix}_{x_col}_vs_{y_col}.html", "interactive")

    def generate_boxplot(self, df: pd.DataFrame, column: str, by_col: str = None, 
                       filename_prefix: str = "boxplot", color: str = None) -> dict:
        """Generates a box plot for a numerical column, optionally grouped by a categorical column."""
        if not self._validate_data(df) or not self._validate_column(df, column):
            return None
        
        # Plotly version
        if by_col and by_col in df.columns and (pd.api.types.is_categorical_dtype(df[by_col]) or pd.api.types.is_object_dtype(df[by_col])):
            fig = px.box(
                df, 
                x=by_col, 
                y=column, 
                title=f'{column} by {by_col}',
                color=by_col,
                color_discrete_sequence=self.color_sequence,
                template=PLOTLY_THEME
            )
        else:
            fig = px.box(
                df, 
                y=column, 
                title=f'Box plot of {column}',
                color_discrete_sequence=[self.color_sequence[0]],
                template=PLOTLY_THEME
            )
        fig.update_layout(
            xaxis_title=by_col if by_col else '',
            yaxis_title=column,
            boxmode='group'
        )
        fig = self._apply_color_customization(fig, color)
        return self._save_plot(fig, f"{filename_prefix}_{column}.html", "interactive")

    def generate_barplot(self, df: pd.DataFrame, x_col: str, y_col: str, top_n: int = None, 
                       filename_prefix: str = "barplot", color: str = None) -> dict:
        """Generates a bar plot for a categorical X and numerical Y column with top N filtering."""
        if not self._validate_data(df) or not self._validate_column(df, x_col) or (y_col and not self._validate_column(df, y_col)):
            return None
        
        # Apply top_n filtering
        if top_n and top_n > 0:
            # Get top N categories by average y_col value
            if y_col:
                top_categories = df.groupby(x_col)[y_col].mean().nlargest(top_n).index
            else:
                top_categories = df[x_col].value_counts().nlargest(top_n).index
            df = df[df[x_col].isin(top_categories)]
        
        # Plotly version
        fig = px.bar(
            df, 
            x=x_col, 
            y=y_col if y_col else None,
            title=f'{"Top " + str(top_n) + " " if top_n else ""}{y_col if y_col else x_col} by {x_col}',
            color=x_col,
            color_discrete_sequence=self.color_sequence,
            template=PLOTLY_THEME
        )
        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=f'Average {y_col}' if y_col else 'Count',
            xaxis_tickangle=-45
        )
        fig = self._apply_color_customization(fig, color)
        return self._save_plot(fig, f"{filename_prefix}_{x_col}_vs_{y_col if y_col else 'count'}.html", "interactive")

    def generate_heatmap(self, df: pd.DataFrame, columns: list = None, filename_prefix: str = "heatmap") -> dict:
        """Generates a correlation heatmap for numerical columns."""
        if columns:
            df_for_corr = df[columns].select_dtypes(include=['number'])
        else:
            df_for_corr = df.select_dtypes(include=['number'])

        if not self._validate_data(df_for_corr) or len(df_for_corr.columns) < 2:
            return None
        
        corr = df_for_corr.corr(numeric_only=True)
        
        # Plotly version
        fig = ff.create_annotated_heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            annotation_text=corr.round(2).values,
            colorscale='RdBu_r',
            zmid=0
        )
        fig.update_layout(
            title='Correlation Heatmap',
            xaxis_title='Columns',
            yaxis_title='Columns',
            template=PLOTLY_THEME
        )
        return self._save_plot(fig, f"{filename_prefix}_correlation.html", "interactive")

    def generate_pairplot(self, df: pd.DataFrame, columns: list, hue_col: str = None, 
                        filename_prefix: str = "pairplot", color: str = None) -> dict:
        """Generates a pairplot for multiple numerical columns."""
        if not self._validate_data(df) or not columns or len(columns) < 2:
            return None
        
        # Plotly version
        fig = px.scatter_matrix(
            df,
            dimensions=columns,
            color=hue_col,
            title="Pairwise Relationships",
            template=PLOTLY_THEME,
            opacity=0.7,
            color_discrete_sequence=self.color_sequence
        )
        fig.update_traces(diagonal_visible=False)
        fig = self._apply_color_customization(fig, color)
        return self._save_plot(fig, f"{filename_prefix}_matrix.html", "interactive")

    def generate_lineplot(self, df: pd.DataFrame, x_col: str, y_col: str, hue_col: str = None, 
                        filename_prefix: str = "lineplot", color: str = None) -> dict:
        """Generates a line plot for temporal data."""
        if not self._validate_data(df) or not self._validate_column(df, x_col) or not self._validate_column(df, y_col):
            return None
        
        # Plotly version
        fig = px.line(
            df, 
            x=x_col, 
            y=y_col, 
            color=hue_col,
            title=f'{y_col} over {x_col}',
            template=PLOTLY_THEME,
            markers=True,
            color_discrete_sequence=self.color_sequence
        )
        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_col,
            hovermode='x unified'
        )
        fig = self._apply_color_customization(fig, color)
        return self._save_plot(fig, f"{filename_prefix}_{x_col}_vs_{y_col}.html", "interactive")

    def generate_violinplot(self, df: pd.DataFrame, column: str, by_col: str = None, 
                          filename_prefix: str = "violinplot", color: str = None) -> dict:
        """Generates a violin plot for a numerical column."""
        if not self._validate_data(df) or not self._validate_column(df, column):
            return None
        
        # Plotly version
        fig = px.violin(
            df, 
            y=column, 
            x=by_col,
            box=True,
            points="all",
            title=f'Distribution of {column}',
            color=by_col,
            template=PLOTLY_THEME,
            color_discrete_sequence=self.color_sequence
        )
        fig.update_layout(
            xaxis_title=by_col if by_col else '',
            yaxis_title=column
        )
        fig = self._apply_color_customization(fig, color)
        return self._save_plot(fig, f"{filename_prefix}_{column}.html", "interactive")

    def generate_piechart(self, df: pd.DataFrame, column: str, top_n: int = None, 
                        filename_prefix: str = "piechart", color: str = None) -> dict:
        """Generates a pie chart for a categorical column with top N filtering."""
        if not self._validate_data(df) or not self._validate_column(df, column):
            return None
        
        value_counts = df[column].value_counts().reset_index()
        value_counts.columns = [column, 'count']
        
        # Apply top_n filtering
        if top_n and top_n > 0:
            value_counts = value_counts.head(top_n)
        
        # Plotly version
        fig = px.pie(
            value_counts, 
            names=column, 
            values='count',
            title=f'{"Top " + str(top_n) + " " if top_n else ""}Distribution of {column}',
            hole=0.3,
            color_discrete_sequence=self.color_sequence,
            template=PLOTLY_THEME
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig = self._apply_color_customization(fig, color)
        return self._save_plot(fig, f"{filename_prefix}_{column}.html", "interactive")

    def generate_qqplot(self, df: pd.DataFrame, column: str, filename_prefix: str = "qqplot") -> dict:
        """Generates a Q-Q plot to check normality."""
        if not self._validate_data(df) or not self._validate_column(df, column) or not pd.api.types.is_numeric_dtype(df[column]):
            return None
        
        # Filter out missing values
        data = df[column].dropna()
        
        # Create Q-Q plot
        qq = stats.probplot(data, dist="norm")
        x = qq[0][0]
        y = qq[0][1]
        
        # Create figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=y, 
            mode='markers', 
            name='Data',
            marker=dict(color=self.color_sequence[0])
        ))
        
        # Add reference line
        fig.add_trace(go.Scatter(
            x=x, y=qq[1][0] + qq[1][1]*x, 
            mode='lines', 
            name='Normal Reference',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=f'Q-Q Plot of {column}',
            xaxis_title='Theoretical Quantiles',
            yaxis_title='Ordered Values',
            showlegend=True,
            template=PLOTLY_THEME
        )
        
        return self._save_plot(fig, f"{filename_prefix}_{column}.html", "interactive")

    def generate_custom_plot(self, df: pd.DataFrame, plot_type: str, params: dict, 
                            filename_prefix: str = "custom_plot") -> dict:
        """Generates a plot based on flexible parameters"""
        plot_funcs = {
            'histogram': self.generate_histogram,
            'scatter': self.generate_scatterplot,
            'box': self.generate_boxplot,
            'bar': self.generate_barplot,
            'heatmap': self.generate_heatmap,
            'pairplot': self.generate_pairplot,
            'line': self.generate_lineplot,
            'violin': self.generate_violinplot,
            'pie': self.generate_piechart,
            'qq': self.generate_qqplot
        }
        
        if plot_type not in plot_funcs:
            logging.error(f"Unsupported plot type: {plot_type}")
            return None
            
        try:
            return plot_funcs[plot_type](df, **params, filename_prefix=filename_prefix)
        except Exception as e:
            logging.error(f"Error generating custom plot: {e}")
            return None