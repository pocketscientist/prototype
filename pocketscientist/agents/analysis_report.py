"""
Analysis Report agent for CRISP-DM workflow.
"""

from typing import Dict, Any
from .base import BaseAgent, AgentState


class AnalysisReportAgent(BaseAgent):
    """Agent responsible for creating comprehensive analysis reports and summaries."""
    
    def __init__(self, llm_provider, executor=None):
        super().__init__(
            name="Analysis Report Specialist",
            llm_provider=llm_provider,
            phase_name="create comprehensive analysis report and summary of findings",
            executor=executor
        )
    
    def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        Execute analysis report phase.
        """
        try:
            # Add phase header
            header_cell = self._add_markdown_cell(
                "# Analysis Report & Summary\n\n"
                "Comprehensive summary of all findings, insights, and conclusions from the complete analysis."
            )
            
            notebook_cells = [header_cell]
            
            # Executive summary of findings
            executive_summary_cell = self._generate_code_cell(
                f"""Generate code to create an executive summary:
1. Summarize the key findings from each analysis phase
2. Answer the original business question: "{state.user_context}"
3. Highlight the most significant insights discovered
4. Create summary visualizations of key findings
5. Provide clear, actionable conclusions

Focus on the most important discoveries that address the user's question.""",
                state
            )
            notebook_cells.append(executive_summary_cell)
            
            # Detailed findings by phase
            detailed_findings_cell = self._add_markdown_cell(
                self._generate_detailed_findings_summary(state)
            )
            notebook_cells.append(detailed_findings_cell)
            
            # Key insights and patterns
            insights_cell = self._generate_code_cell(
                """Generate code to highlight key insights and patterns:
1. Identify the most important patterns found in the data
2. Create visualizations that showcase key insights
3. Quantify the significance of findings
4. Compare different aspects of the analysis
5. Create a visual summary of discoveries

Make insights clear and visually compelling.""",
                state
            )
            notebook_cells.append(insights_cell)
            
            # Data quality and methodology summary
            methodology_cell = self._add_markdown_cell(
                self._generate_methodology_summary(state)
            )
            notebook_cells.append(methodology_cell)
            
            # Conclusions and next steps
            conclusions_cell = self._generate_code_cell(
                f"""Generate code to create final conclusions:
1. Answer the original question: "{state.user_context}"
2. Provide clear, evidence-based conclusions
3. Quantify confidence levels in findings
4. Suggest areas for further investigation
5. Create a final summary table of key metrics/findings

Focus on providing definitive answers based on the analysis performed.""",
                state
            )
            notebook_cells.append(conclusions_cell)
            
            # Final report export
            export_cell = self._generate_code_cell(
                """Generate code for report documentation:
1. Create a comprehensive summary of all findings
2. Export key visualizations as high-quality images
3. Generate a findings summary table
4. Save important datasets or results
5. Create a final analysis report document

Prepare all materials for stakeholders and documentation.""",
                state
            )
            notebook_cells.append(export_cell)
            
            findings = {
                "analysis_report_complete": True,
                "executive_summary_created": True,
                "detailed_findings_documented": True,
                "key_insights_identified": True,
                "methodology_documented": True,
                "conclusions_provided": True,
                "final_answer": self._extract_final_answer(state)
            }
            
            return {
                "success": True,
                "findings": findings,
                "notebook_cells": notebook_cells,
                "next_phase": None,  # Final phase
                "message": "Analysis report completed - comprehensive summary of all findings ready",
                "requires_iteration": False
            }
            
        except Exception as e:
            return {
                "success": False,
                "findings": {"error": str(e)},
                "notebook_cells": [],
                "next_phase": None,  # Still final phase
                "message": f"Analysis report failed: {str(e)}",
                "requires_iteration": False
            }
    
    def _generate_detailed_findings_summary(self, state: AgentState) -> str:
        """Generate detailed summary of findings from each phase."""
        
        summary_parts = ["## Detailed Findings by Phase\n"]
        
        # Iterate through completed phases and their findings
        for phase in state.completed_phases:
            if phase in state.findings:
                phase_findings = state.findings[phase]
                phase_name = phase.replace('_', ' ').title()
                
                summary_parts.append(f"### {phase_name}\n")
                
                # Add key findings for this phase
                if isinstance(phase_findings, dict):
                    for key, value in phase_findings.items():
                        if isinstance(value, bool) and value:
                            summary_parts.append(f"- ✅ {key.replace('_', ' ').title()}")
                        elif isinstance(value, str):
                            summary_parts.append(f"- **{key.replace('_', ' ').title()}**: {value}")
                        elif isinstance(value, (list, dict)) and value:
                            summary_parts.append(f"- **{key.replace('_', ' ').title()}**: Available")
                
                summary_parts.append("")  # Empty line between phases
        
        # Add execution summary
        if state.execution_results:
            summary_parts.append("### Execution Summary\n")
            successful_executions = sum(1 for result in state.execution_results.values() 
                                      if result.get("success", False))
            total_executions = len(state.execution_results)
            summary_parts.append(f"- **Code Executions**: {successful_executions}/{total_executions} successful")
            
            if state.useful_patterns:
                summary_parts.append(f"- **Patterns Discovered**: {len(state.useful_patterns)}")
            
            summary_parts.append("")
        
        return "\n".join(summary_parts)
    
    def _generate_methodology_summary(self, state: AgentState) -> str:
        """Generate summary of methodology and data quality."""
        
        methodology_parts = [
            "## Methodology & Data Quality Summary\n",
            "### Analysis Approach",
            f"- **Dataset**: {state.dataset_path}",
            f"- **Analysis Question**: {state.user_context}",
            f"- **Phases Completed**: {', '.join(sorted(state.completed_phases))}",
            f"- **Total Execution Time**: {(state.iteration_count)} iterations",
            ""
        ]
        
        # Add data quality insights if available
        data_findings = state.findings.get('data_understanding', {})
        if data_findings:
            methodology_parts.extend([
                "### Data Quality",
                "- Data loading and exploration completed",
                "- Quality assessment performed",
                "- Initial patterns identified",
                ""
            ])
        
        # Add preparation insights if available
        prep_findings = state.findings.get('data_preparation', {})
        if prep_findings:
            methodology_parts.extend([
                "### Data Preparation",
                "- Data cleaning and transformation completed",
                "- Feature engineering performed (if applicable)",
                "- Data validation conducted",
                ""
            ])
        
        # Add modeling insights if available
        model_findings = state.findings.get('modeling', {})
        if model_findings:
            methodology_parts.extend([
                "### Modeling Approach",
                f"- **Approach**: {model_findings.get('approach', 'Not specified')}",
                f"- **Algorithms Used**: {', '.join(model_findings.get('algorithms_used', []))}",
                "- Model performance evaluated",
                ""
            ])
        
        # Add limitations and notes
        methodology_parts.extend([
            "### Limitations & Notes",
            "- Analysis based on provided dataset only",
            "- Findings are specific to the data and context analyzed",
            "- Results should be validated with domain expertise",
            ""
        ])
        
        return "\n".join(methodology_parts)
    
    def _extract_final_answer(self, state: AgentState) -> str:
        """Extract a concise final answer to the user's question."""
        
        # Try to create a concise answer based on what was discovered
        context_lower = state.user_context.lower()
        
        if "interesting" in context_lower or "patterns" in context_lower:
            return "Key patterns and insights identified in the data analysis"
        elif any(word in context_lower for word in ["predict", "forecast", "model"]):
            return "Predictive models built and evaluated for the specified requirements"
        elif any(word in context_lower for word in ["improve", "optimize", "better"]):
            return "Optimization opportunities and improvement recommendations identified"
        else:
            return f"Analysis completed addressing: {state.user_context}"