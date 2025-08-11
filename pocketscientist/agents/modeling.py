"""
Modeling agent for CRISP-DM workflow.
"""

from typing import Dict, Any
from .base import BaseAgent, AgentState


class ModelingAgent(BaseAgent):
    """Agent responsible for selecting and applying modeling techniques."""
    
    def __init__(self, llm_provider, executor=None):
        super().__init__(
            name="Modeling Specialist",
            llm_provider=llm_provider,
            phase_name="select and apply appropriate modeling techniques",
            executor=executor
        )
    
    def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        Execute modeling phase.
        """
        try:
            # Determine modeling approach based on context
            modeling_strategy = self._determine_modeling_strategy(state)
            
            # Add phase header
            header_cell = self._add_markdown_cell(
                f"# Modeling\n\n"
                f"**Approach:** {modeling_strategy['approach']}\n\n"
                f"**Algorithms:** {', '.join(modeling_strategy['algorithms'])}\n\n"
                f"**Objective:** {modeling_strategy['objective']}"
            )
            
            notebook_cells = [header_cell]
            
            # Problem setup
            setup_cell = self._generate_code_cell(
                f"""Generate code to set up the modeling problem:
1. Define the target variable based on the user's request: "{state.user_context}"
2. Select relevant features for the model
3. Split data into training and testing sets (if appropriate)
4. Handle any final preprocessing specific to modeling
5. Set up the modeling framework

Problem type: {modeling_strategy['approach']}
Target: Based on user context about "{state.user_context}"
""",
                state
            )
            notebook_cells.append(setup_cell)
            
            # Model building
            for algorithm in modeling_strategy['algorithms']:
                model_cell = self._generate_code_cell(
                    f"""Generate code to build and train a {algorithm} model:
1. Initialize the {algorithm} algorithm with appropriate parameters
2. Train the model on the training data
3. Make predictions on the test set
4. Calculate relevant performance metrics
5. Display model results and interpretation

Focus on {modeling_strategy['approach']} performance metrics.""",
                    state
                )
                notebook_cells.append(model_cell)
            
            # Model comparison
            if len(modeling_strategy['algorithms']) > 1:
                comparison_cell = self._generate_code_cell(
                    f"""Generate code to compare model performance:
1. Create a comparison table of all models tried
2. Visualize model performance using appropriate plots
3. Identify the best performing model
4. Explain the trade-offs between different approaches
5. Select the final recommended model

Compare: {', '.join(modeling_strategy['algorithms'])}""",
                    state
                )
                notebook_cells.append(comparison_cell)
            
            # Model interpretation
            interpretation_cell = self._generate_code_cell(
                """Generate code for model interpretation:
1. Show feature importance or model coefficients
2. Create visualizations explaining model behavior
3. Generate example predictions with explanations
4. Identify key factors driving predictions
5. Discuss model limitations and assumptions

Make the model results understandable and actionable.""",
                state
            )
            notebook_cells.append(interpretation_cell)
            
            findings = {
                "modeling_complete": True,
                "approach": modeling_strategy['approach'],
                "algorithms_used": modeling_strategy['algorithms'],
                "models_built": True,
                "performance_evaluated": True
            }
            
            return {
                "success": True,
                "findings": findings,
                "notebook_cells": notebook_cells,
                "next_phase": "evaluation",
                "message": f"Modeling completed using {modeling_strategy['approach']} approach with {len(modeling_strategy['algorithms'])} algorithms",
                "requires_iteration": False
            }
            
        except Exception as e:
            return {
                "success": False,
                "findings": {"error": str(e)},
                "notebook_cells": [],
                "next_phase": "evaluation",  # Continue anyway
                "message": f"Modeling failed: {str(e)}",
                "requires_iteration": False
            }
    
    def _determine_modeling_strategy(self, state: AgentState) -> Dict[str, Any]:
        """Determine the appropriate modeling strategy based on user context."""
        
        context_lower = state.user_context.lower()
        
        # Regression keywords
        regression_keywords = [
            "predict", "forecast", "estimate", "price", "value", "amount", 
            "continuous", "numerical", "regression", "how much", "what will"
        ]
        
        # Classification keywords  
        classification_keywords = [
            "classify", "category", "type", "class", "binary", "yes/no",
            "classification", "identify", "detect", "which", "what kind"
        ]
        
        # Clustering keywords
        clustering_keywords = [
            "segment", "group", "cluster", "similar", "patterns", "categories",
            "unsupervised", "groups", "types"
        ]
        
        # Determine problem type
        if any(word in context_lower for word in regression_keywords):
            return {
                "approach": "regression",
                "algorithms": ["Linear Regression", "Random Forest Regressor", "XGBoost Regressor"],
                "objective": "Predict continuous numerical values"
            }
        elif any(word in context_lower for word in classification_keywords):
            return {
                "approach": "classification", 
                "algorithms": ["Logistic Regression", "Random Forest Classifier", "XGBoost Classifier"],
                "objective": "Predict categorical outcomes"
            }
        elif any(word in context_lower for word in clustering_keywords):
            return {
                "approach": "clustering",
                "algorithms": ["K-Means", "DBSCAN", "Hierarchical Clustering"],
                "objective": "Discover hidden patterns and group similar data points"
            }
        else:
            # Default: try both regression and classification
            return {
                "approach": "exploratory modeling",
                "algorithms": ["Random Forest", "Logistic Regression"],
                "objective": "Explore predictive relationships in the data"
            }