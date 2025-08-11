"""
Evaluation agent for CRISP-DM workflow.
"""

from typing import Dict, Any
from .base import BaseAgent, AgentState


class EvaluationAgent(BaseAgent):
    """Agent responsible for evaluating model performance and validating results."""
    
    def __init__(self, llm_provider, executor=None):
        super().__init__(
            name="Model Evaluation Specialist",
            llm_provider=llm_provider,
            phase_name="evaluate model performance and validate results",
            executor=executor
        )
    
    def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        Execute evaluation phase.
        """
        try:
            # Add phase header
            header_cell = self._add_markdown_cell(
                "# Model Evaluation\n\n"
                "Comprehensive evaluation of model performance and validation of results."
            )
            
            notebook_cells = [header_cell]
            
            # Performance evaluation
            performance_cell = self._generate_code_cell(
                """Generate code for comprehensive model performance evaluation:
1. Calculate detailed performance metrics appropriate for the problem type
2. Create confusion matrices, ROC curves, or other relevant evaluation plots
3. Perform cross-validation to assess model stability
4. Test for overfitting or underfitting
5. Analyze residuals or prediction errors

Provide clear interpretation of what these metrics mean for the business problem.""",
                state
            )
            notebook_cells.append(performance_cell)
            
            # Model validation
            validation_cell = self._generate_code_cell(
                """Generate code for model validation:
1. Test model assumptions (if applicable)
2. Check for data leakage or other validation issues
3. Assess model fairness and bias
4. Test model robustness with different data subsets
5. Validate that the model answers the original business question

Ensure the model is reliable and trustworthy.""",
                state
            )
            notebook_cells.append(validation_cell)
            
            # Business impact assessment
            impact_cell = self._generate_code_cell(
                f"""Generate code to assess business impact:
1. Translate model performance into business terms
2. Calculate potential business value or cost savings
3. Assess practical significance vs statistical significance
4. Identify limitations and risks of using this model
5. Provide recommendations for model deployment

Consider the original user question: "{state.user_context}"
""",
                state
            )
            notebook_cells.append(impact_cell)
            
            # Model comparison and selection
            selection_cell = self._generate_code_cell(
                """Generate code for final model selection:
1. Compare all models built in terms of performance and interpretability
2. Select the best model considering business constraints
3. Document the rationale for model selection
4. Prepare model summary and key insights
5. Create final model evaluation report

Balance performance, interpretability, and business requirements.""",
                state
            )
            notebook_cells.append(selection_cell)
            
            findings = {
                "evaluation_complete": True,
                "performance_assessed": True,
                "model_validated": True,
                "business_impact_evaluated": True,
                "final_model_selected": True
            }
            
            return {
                "success": True,
                "findings": findings,
                "notebook_cells": notebook_cells,
                "next_phase": "analysis_report",
                "message": "Model evaluation completed - ready for deployment preparation",
                "requires_iteration": False
            }
            
        except Exception as e:
            return {
                "success": False,
                "findings": {"error": str(e)},
                "notebook_cells": [],
                "next_phase": "analysis_report",  # Continue anyway
                "message": f"Evaluation failed: {str(e)}",
                "requires_iteration": False
            }