"""
Business Understanding agent for CRISP-DM workflow.
"""

from typing import Dict, Any
from .base import BaseAgent, AgentState


class BusinessUnderstandingAgent(BaseAgent):
    """Agent responsible for understanding business objectives and requirements."""
    
    def __init__(self, llm_provider):
        super().__init__(
            name="Business Understanding Specialist", 
            llm_provider=llm_provider,
            phase_name="define business objectives and project requirements"
        )
    
    def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        Execute business understanding phase.
        """
        try:
            # Add phase header
            header_cell = self._add_markdown_cell(
                f"# Business Understanding\n\n"
                f"**User Context:** {state.user_context}\n\n"
                f"**Objective:** Define the business problem and success criteria"
            )
            
            # Analyze the user context and define objectives
            objectives = self._analyze_business_context(state)
            
            # Create notebook cells
            notebook_cells = [header_cell]
            
            # Add analysis cell
            analysis_cell = self._add_markdown_cell(
                f"## Analysis Objectives\n\n"
                f"{objectives['analysis_summary']}\n\n"
                f"**Success Criteria:**\n"
                f"{objectives['success_criteria']}\n\n"
                f"**Key Questions:**\n"
                f"{objectives['key_questions']}"
            )
            notebook_cells.append(analysis_cell)
            
            # Add setup code cell
            setup_cell = self._generate_code_cell(
                "Generate Python code to set up the data science environment. "
                "Include necessary imports for pandas, numpy, matplotlib, seaborn, sklearn, etc. "
                "Set up plotting parameters and any other common setup code."
            )
            notebook_cells.append(setup_cell)
            
            findings = {
                "objectives_defined": True,
                "analysis_type": objectives["analysis_type"],
                "success_criteria": objectives["success_criteria"],
                "key_questions": objectives["key_questions"],
                "recommended_approach": objectives["approach"]
            }
            
            return {
                "success": True,
                "findings": findings,
                "notebook_cells": notebook_cells,
                "next_phase": "data_understanding",
                "message": f"Business objectives defined: {objectives['analysis_type']} analysis",
                "requires_iteration": False
            }
            
        except Exception as e:
            return {
                "success": False,
                "findings": {"error": str(e)},
                "notebook_cells": [],
                "next_phase": "data_understanding",  # Continue anyway
                "message": f"Business understanding failed: {str(e)}",
                "requires_iteration": False
            }
    
    def _analyze_business_context(self, state: AgentState) -> Dict[str, Any]:
        """Analyze the user context to extract business objectives."""
        
        prompt = f"""
Analyze this user request and define clear business objectives for a data science project:

User Context: "{state.user_context}"
Dataset: {state.dataset_path}

Based on the user's request, determine:
1. What type of analysis is needed? (descriptive, diagnostic, predictive, prescriptive)
2. What are the key business questions to answer?
3. What would constitute success for this analysis?
4. What approach should we take?

Respond with a structured analysis that includes:
- Analysis type and summary
- Success criteria (specific and measurable)
- 3-5 key questions to investigate
- Recommended analytical approach

Be specific and actionable. Focus on what the user actually wants to learn from their data.
"""
        
        try:
            response = self.llm_provider.generate(
                prompt=prompt,
                system_prompt=self._get_system_prompt(),
                temperature=0.5
            )
            
            # Parse the response into structured format
            return self._parse_business_analysis(response, state.user_context)
            
        except Exception as e:
            # Fallback analysis
            return self._fallback_business_analysis(state.user_context)
    
    def _parse_business_analysis(self, response: str, user_context: str) -> Dict[str, Any]:
        """Parse LLM response into structured business analysis."""
        
        # Try to extract structured information from the response
        lines = response.strip().split('\n')
        
        analysis_type = "exploratory"  # Default
        analysis_summary = response[:300] + "..." if len(response) > 300 else response
        
        # Look for key indicators in user context
        context_lower = user_context.lower()
        if any(word in context_lower for word in ["predict", "forecast", "estimate"]):
            analysis_type = "predictive"
        elif any(word in context_lower for word in ["why", "cause", "reason", "explain"]):
            analysis_type = "diagnostic"  
        elif any(word in context_lower for word in ["recommend", "optimize", "should", "best"]):
            analysis_type = "prescriptive"
        elif any(word in context_lower for word in ["describe", "summarize", "overview", "interesting"]):
            analysis_type = "descriptive"
        
        return {
            "analysis_type": analysis_type,
            "analysis_summary": analysis_summary,
            "success_criteria": f"Provide clear insights that address: {user_context}",
            "key_questions": f"Questions derived from: {user_context}",
            "approach": f"Data-driven {analysis_type} analysis approach"
        }
    
    def _fallback_business_analysis(self, user_context: str) -> Dict[str, Any]:
        """Fallback business analysis when LLM fails."""
        return {
            "analysis_type": "exploratory",
            "analysis_summary": f"Exploratory analysis based on user request: {user_context}",
            "success_criteria": "Discover meaningful patterns and insights in the data",
            "key_questions": f"Investigate: {user_context}",
            "approach": "Systematic data exploration and analysis"
        }