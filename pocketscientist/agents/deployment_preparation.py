"""
Deployment Preparation agent for CRISP-DM workflow.
"""

from typing import Dict, Any
from .base import BaseAgent, AgentState


class DeploymentPreparationAgent(BaseAgent):
    """Agent responsible for preparing final recommendations and actionable insights."""
    
    def __init__(self, llm_provider):
        super().__init__(
            name="Deployment Preparation Specialist",
            llm_provider=llm_provider,
            phase_name="prepare final recommendations and actionable insights"
        )
    
    def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        Execute deployment preparation phase.
        """
        try:
            # Add phase header
            header_cell = self._add_markdown_cell(
                "# Deployment & Final Recommendations\n\n"
                "Summary of findings and actionable recommendations based on the analysis."
            )
            
            notebook_cells = [header_cell]
            
            # Executive summary
            summary_cell = self._generate_code_cell(
                f"""Generate code to create an executive summary:
1. Summarize the key findings from the entire analysis
2. Answer the original business question: "{state.user_context}"
3. Highlight the most important insights discovered
4. Quantify the impact or significance of findings
5. Create executive-level visualizations

Focus on clear, non-technical language that stakeholders can understand."""
            )
            notebook_cells.append(summary_cell)
            
            # Actionable recommendations
            recommendations_cell = self._add_markdown_cell(
                self._generate_recommendations(state)
            )
            notebook_cells.append(recommendations_cell)
            
            # Implementation roadmap
            roadmap_cell = self._add_markdown_cell(
                self._generate_implementation_roadmap(state)
            )
            notebook_cells.append(roadmap_cell)
            
            # Risk assessment and limitations
            risks_cell = self._add_markdown_cell(
                "## Risk Assessment & Limitations\n\n"
                "### Limitations of This Analysis\n"
                "- Data limitations and assumptions made\n"
                "- Model limitations and confidence intervals\n"
                "- Scope and generalizability constraints\n\n"
                "### Risks and Mitigation Strategies\n"
                "- Potential risks in implementing recommendations\n"
                "- Monitoring and validation strategies\n"
                "- Contingency plans\n\n"
                "### Areas for Future Investigation\n"
                "- Additional data that could improve analysis\n"
                "- Follow-up questions to explore\n"
                "- Potential model improvements"
            )
            notebook_cells.append(risks_cell)
            
            # Final validation and export
            export_cell = self._generate_code_cell(
                """Generate code for final deliverables:
1. Create final summary statistics and key metrics
2. Export important datasets or model results
3. Save all key visualizations as high-quality images
4. Generate a data dictionary or documentation
5. Create a final summary report

Prepare all materials needed for stakeholders."""
            )
            notebook_cells.append(export_cell)
            
            findings = {
                "deployment_ready": True,
                "recommendations_prepared": True,
                "executive_summary_created": True,
                "implementation_roadmap_defined": True,
                "risks_assessed": True,
                "deliverables_prepared": True
            }
            
            return {
                "success": True,
                "findings": findings,
                "notebook_cells": notebook_cells,
                "next_phase": None,  # Final phase
                "message": "Deployment preparation completed - analysis ready for stakeholders",
                "requires_iteration": False
            }
            
        except Exception as e:
            return {
                "success": False,
                "findings": {"error": str(e)},
                "notebook_cells": [],
                "next_phase": None,  # Still final phase
                "message": f"Deployment preparation failed: {str(e)}",
                "requires_iteration": False
            }
    
    def _generate_recommendations(self, state: AgentState) -> str:
        """Generate actionable recommendations based on analysis findings."""
        
        recommendations = []
        
        # Based on user context, generate relevant recommendations
        context_lower = state.user_context.lower()
        
        if "interesting" in context_lower or "patterns" in context_lower:
            recommendations.extend([
                "**Investigate Key Patterns**: Follow up on the most significant patterns discovered",
                "**Data Collection**: Consider collecting additional data in areas of interest",
                "**Regular Monitoring**: Set up processes to track these patterns over time"
            ])
        
        if any(word in context_lower for word in ["predict", "forecast", "model"]):
            recommendations.extend([
                "**Model Deployment**: Implement the validated model in a production environment",
                "**Performance Monitoring**: Set up automated monitoring of model performance",
                "**Regular Retraining**: Establish a schedule for model updates and retraining",
                "**A/B Testing**: Consider testing model predictions against current methods"
            ])
        
        if any(word in context_lower for word in ["improve", "optimize", "better"]):
            recommendations.extend([
                "**Process Improvement**: Implement changes based on analysis findings",
                "**Pilot Program**: Start with a small-scale implementation to test recommendations",
                "**Success Metrics**: Define clear KPIs to measure improvement impact"
            ])
        
        # Default recommendations if none above apply
        if not recommendations:
            recommendations = [
                "**Data-Driven Decisions**: Use insights from this analysis to inform decision making",
                "**Continuous Monitoring**: Track key metrics identified in this analysis",
                "**Further Analysis**: Consider deeper investigation into significant findings"
            ]
        
        return "## Actionable Recommendations\n\n" + "\n".join(f"- {rec}" for rec in recommendations)
    
    def _generate_implementation_roadmap(self, state: AgentState) -> str:
        """Generate an implementation roadmap."""
        
        return """## Implementation Roadmap

### Immediate Actions (0-30 days)
- Review and validate analysis findings with stakeholders
- Identify resource requirements for implementation
- Set up data infrastructure and monitoring systems

### Short-term Implementation (1-3 months)
- Begin pilot implementation of key recommendations
- Establish baseline measurements and KPIs
- Train relevant team members on new processes or tools

### Long-term Monitoring (3+ months)
- Monitor performance and impact of implemented changes
- Refine and optimize based on real-world results
- Plan for periodic re-analysis and model updates

### Success Criteria
- Define specific, measurable outcomes that indicate success
- Establish regular review checkpoints
- Set up feedback loops for continuous improvement"""