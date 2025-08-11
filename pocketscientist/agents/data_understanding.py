"""
Data Understanding agent for CRISP-DM workflow.
"""

from typing import Dict, Any
from .base import BaseAgent, AgentState


class DataUnderstandingAgent(BaseAgent):
    """Agent responsible for initial data exploration and quality assessment."""
    
    def __init__(self, llm_provider, executor=None):
        super().__init__(
            name="Data Understanding Specialist",
            llm_provider=llm_provider,
            phase_name="explore data structure, quality, and initial patterns",
            executor=executor
        )
    
    def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        Execute data understanding phase.
        """
        try:
            # Add phase header
            header_cell = self._add_markdown_cell(
                "# Data Understanding\n\n"
                "Initial exploration of the dataset to understand structure, quality, and characteristics."
            )
            
            notebook_cells = [header_cell]
            
            # Data loading and basic info
            load_cell = self._generate_code_cell(
                f"""Generate code to:
1. Load the dataset from '{state.dataset_path}' using pandas
2. Display basic information about the dataset (shape, dtypes, info)
3. Show the first few rows
4. Display summary statistics

Handle potential encoding issues and provide informative output.""",
                state
            )
            notebook_cells.append(load_cell)
            
            # Data quality assessment
            quality_cell = self._generate_code_cell(
                """Generate code to assess data quality:
1. Check for missing values (count and percentage)
2. Identify duplicate rows
3. Look for potential outliers in numerical columns
4. Check for inconsistent data types
5. Analyze categorical variables (unique values, frequency)

Create visualizations where appropriate and summarize findings.""",
                state
            )
            notebook_cells.append(quality_cell)
            
            # Initial visualizations
            viz_cell = self._generate_code_cell(
                """Generate code for initial data visualization:
1. Create distribution plots for numerical variables
2. Create bar plots for categorical variables (top categories)
3. Create a correlation matrix heatmap if applicable
4. Generate any other relevant exploratory plots
5. Save important plots as PNG files

Focus on understanding data patterns and relationships.""",
                state
            )
            notebook_cells.append(viz_cell)
            
            # Data insights summary
            insights_cell = self._generate_code_cell(
                """Generate code to summarize key data insights:
1. Identify the most important patterns found
2. Note any data quality issues that need addressing
3. Highlight interesting relationships or anomalies
4. Suggest areas for further investigation
5. Create a summary report of findings

Print clear, actionable insights.""",
                state
            )
            notebook_cells.append(insights_cell)
            
            findings = {
                "data_exploration_complete": True,
                "data_loaded": True,
                "quality_assessed": True,
                "initial_patterns_identified": True
            }
            
            return {
                "success": True,
                "findings": findings,
                "notebook_cells": notebook_cells,
                "next_phase": "data_preparation",
                "message": "Data understanding completed - ready for preparation phase",
                "requires_iteration": False
            }
            
        except Exception as e:
            return {
                "success": False,
                "findings": {"error": str(e)},
                "notebook_cells": [],
                "next_phase": "data_preparation",  # Continue anyway
                "message": f"Data understanding failed: {str(e)}",
                "requires_iteration": False
            }