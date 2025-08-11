"""
Data Preparation agent for CRISP-DM workflow.
"""

from typing import Dict, Any
from .base import BaseAgent, AgentState


class DataPreparationAgent(BaseAgent):
    """Agent responsible for data cleaning, transformation, and feature engineering."""
    
    def __init__(self, llm_provider, executor=None):
        super().__init__(
            name="Data Preparation Specialist",
            llm_provider=llm_provider, 
            phase_name="clean, transform, and prepare data for analysis",
            executor=executor
        )
    
    def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        Execute data preparation phase.
        """
        try:
            # Determine what preparation is needed based on user context and previous findings
            prep_strategy = self._determine_preparation_strategy(state)
            
            # Add phase header
            header_cell = self._add_markdown_cell(
                f"# Data Preparation\n\n"
                f"**Strategy:** {prep_strategy['description']}\n\n"
                f"**Steps:** {', '.join(prep_strategy['steps'])}"
            )
            
            notebook_cells = [header_cell]
            
            # Data cleaning
            if "cleaning" in prep_strategy['steps']:
                cleaning_cell = self._generate_code_cell(
                    """Generate code for data cleaning:
1. Handle missing values (drop, fill, or impute based on context)
2. Remove or fix duplicate entries
3. Handle outliers (identify and decide whether to keep, transform, or remove)
4. Fix data type inconsistencies
5. Standardize categorical values

Explain your decisions and show before/after comparisons.""",
                    state
                )
                notebook_cells.append(cleaning_cell)
            
            # Data transformation
            if "transformation" in prep_strategy['steps']:
                transform_cell = self._generate_code_cell(
                    """Generate code for data transformation:
1. Normalize or standardize numerical features if needed
2. Encode categorical variables (one-hot, label encoding, etc.)
3. Handle date/time features if present
4. Apply any necessary scaling or normalization
5. Transform skewed distributions if needed

Choose appropriate methods based on the data and analysis goals.""",
                    state
                )
                notebook_cells.append(transform_cell)
            
            # Feature engineering
            if "feature_engineering" in prep_strategy['steps']:
                feature_cell = self._generate_code_cell(
                    """Generate code for feature engineering:
1. Create new features from existing ones (combinations, ratios, etc.)
2. Extract useful information from text or date columns
3. Create binned versions of continuous variables if useful
4. Generate interaction terms if relevant
5. Select most relevant features

Focus on features that might help answer the user's questions.""",
                    state
                )
                notebook_cells.append(feature_cell)
            
            # Validation and summary
            validation_cell = self._generate_code_cell(
                """Generate code to validate the prepared data:
1. Check the final dataset structure and quality
2. Verify no unexpected issues were introduced
3. Compare key statistics before and after preparation
4. Create visualizations showing the impact of preparation steps
5. Save the cleaned dataset if appropriate

Summarize what was accomplished in this preparation phase.""",
                state
            )
            notebook_cells.append(validation_cell)
            
            findings = {
                "data_preparation_complete": True,
                "preparation_strategy": prep_strategy,
                "data_cleaned": "cleaning" in prep_strategy['steps'],
                "data_transformed": "transformation" in prep_strategy['steps'],
                "features_engineered": "feature_engineering" in prep_strategy['steps']
            }
            
            # Decide next phase based on user context
            next_phase = self._determine_next_phase(state)
            
            return {
                "success": True,
                "findings": findings,
                "notebook_cells": notebook_cells,
                "next_phase": next_phase,
                "message": f"Data preparation completed using {prep_strategy['description']} strategy",
                "requires_iteration": False
            }
            
        except Exception as e:
            return {
                "success": False,
                "findings": {"error": str(e)},
                "notebook_cells": [],
                "next_phase": "modeling",  # Continue anyway
                "message": f"Data preparation failed: {str(e)}",
                "requires_iteration": False
            }
    
    def _determine_preparation_strategy(self, state: AgentState) -> Dict[str, Any]:
        """Determine what data preparation steps are needed."""
        
        context_lower = state.user_context.lower()
        
        # Default strategy
        strategy = {
            "description": "Standard data cleaning and preparation",
            "steps": ["cleaning", "transformation"]
        }
        
        # Add feature engineering if modeling is likely needed
        if any(word in context_lower for word in [
            "predict", "model", "classify", "regression", "machine learning", "ml"
        ]):
            strategy["steps"].append("feature_engineering")
            strategy["description"] = "Comprehensive preparation for modeling"
        
        # Simple cleaning if just exploration
        if any(word in context_lower for word in [
            "explore", "describe", "summarize", "overview", "interesting"
        ]) and not any(word in context_lower for word in ["predict", "model"]):
            strategy = {
                "description": "Basic cleaning for exploration",
                "steps": ["cleaning"]
            }
        
        return strategy
    
    def _determine_next_phase(self, state: AgentState) -> str:
        """Determine the next phase based on context."""
        
        context_lower = state.user_context.lower()
        
        # If modeling is explicitly mentioned or implied, go to modeling
        if any(word in context_lower for word in [
            "predict", "model", "classify", "regression", "machine learning", "ml",
            "forecast", "estimate"
        ]):
            return "modeling"
        
        # If just exploration, skip modeling
        if any(word in context_lower for word in [
            "explore", "describe", "summarize", "overview", "interesting", "patterns"
        ]) and not any(word in context_lower for word in ["predict", "model"]):
            return "analysis_report"
        
        # Default: let coordinator decide
        return "coordinator"