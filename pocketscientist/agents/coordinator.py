"""
Coordinator agent for managing CRISP-DM workflow transitions.
"""

from typing import Dict, Any, Optional
from .base import BaseAgent, AgentState


class CoordinatorAgent(BaseAgent):
    """Coordinates the CRISP-DM workflow and decides phase transitions."""
    
    def __init__(self, llm_provider):
        super().__init__(
            name="CRISP-DM Coordinator",
            llm_provider=llm_provider,
            phase_name="coordinate workflow transitions and decisions"
        )
    
    def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        Decide the next phase based on current state and findings.
        """
        state.update_time_remaining()
        
        # Check if we should stop due to time or iteration limits
        if not state.can_continue_iteration():
            return self._prepare_final_phase(state)
        
        # Analyze current state and decide next action
        decision = self._make_phase_decision(state)
        
        return {
            "success": True,
            "findings": {"coordinator_decision": decision},
            "notebook_cells": [],
            "next_phase": decision["next_phase"],
            "message": decision["reasoning"],
            "requires_iteration": decision.get("requires_iteration", False)
        }
    
    def _make_phase_decision(self, state: AgentState) -> Dict[str, Any]:
        """Make a decision about the next phase."""
        
        # Build context for decision making
        context = self._build_decision_context(state)
        
        prompt = f"""
Based on the current analysis state, determine the next best phase to execute.

Current Context:
{context}

CRISP-DM Phases:
1. business_understanding - Define objectives and requirements
2. data_understanding - Initial data exploration and quality assessment
3. data_preparation - Data cleaning, transformation, feature engineering
4. modeling - Select and apply modeling techniques
5. evaluation - Evaluate model performance and validate results
6. deployment_preparation - Prepare final recommendations and insights

Decision Criteria:
- Have we understood the business objective clearly?
- Do we understand the data structure and quality?
- Is the data ready for modeling?
- Have we built and evaluated models?
- Are we ready for final recommendations?
- Is there time remaining: {state.time_remaining:.0f} seconds
- Current iteration: {state.iteration_count}/{state.max_iterations}

Respond with a JSON object containing:
{{
    "next_phase": "phase_name",
    "reasoning": "explanation of why this phase is needed",
    "requires_iteration": true/false,
    "priority": "high/medium/low"
}}

If we've completed enough analysis or are running out of time, recommend "deployment_preparation" as the final phase.
"""
        
        try:
            response = self.llm_provider.generate(
                prompt=prompt,
                system_prompt="You are an expert data science project coordinator. Make logical decisions about CRISP-DM workflow progression.",
                temperature=0.3
            )
            
            # Parse the JSON response
            import json
            decision = json.loads(response.strip())
            
            # Validate the decision
            valid_phases = [
                "business_understanding", 
                "data_understanding",
                "data_preparation", 
                "modeling", 
                "evaluation", 
                "deployment_preparation"
            ]
            
            if decision["next_phase"] not in valid_phases:
                decision["next_phase"] = "deployment_preparation"
            
            return decision
            
        except Exception as e:
            # Fallback decision logic
            return self._fallback_decision(state)
    
    def _build_decision_context(self, state: AgentState) -> str:
        """Build context string for decision making."""
        context_parts = []
        
        context_parts.append(f"User Context: {state.user_context}")
        context_parts.append(f"Current Phase: {state.current_phase}")
        context_parts.append(f"Phase History: {' → '.join(state.phase_history)}")
        context_parts.append(f"Completed Phases: {', '.join(state.completed_phases)}")
        
        # Add recent findings
        if state.findings:
            context_parts.append("\nKey Findings:")
            for phase, findings in state.findings.items():
                if isinstance(findings, dict):
                    context_parts.append(f"  {phase}: {str(findings)[:200]}...")
                else:
                    context_parts.append(f"  {phase}: {str(findings)[:200]}...")
        
        # Add any errors
        if state.errors:
            context_parts.append(f"\nRecent Errors: {'; '.join(state.errors[-3:])}")
        
        return "\n".join(context_parts)
    
    def _fallback_decision(self, state: AgentState) -> Dict[str, Any]:
        """Fallback decision logic when LLM fails."""
        
        # Simple rule-based fallback
        if "business_understanding" not in state.completed_phases:
            return {
                "next_phase": "business_understanding",
                "reasoning": "Need to establish business objectives first",
                "requires_iteration": False,
                "priority": "high"
            }
        elif "data_understanding" not in state.completed_phases:
            return {
                "next_phase": "data_understanding", 
                "reasoning": "Need to explore and understand the data",
                "requires_iteration": False,
                "priority": "high"
            }
        elif "data_preparation" not in state.completed_phases:
            return {
                "next_phase": "data_preparation",
                "reasoning": "Need to prepare data for analysis",
                "requires_iteration": False,
                "priority": "high"
            }
        elif "modeling" not in state.completed_phases and "modeling" in state.user_context.lower():
            return {
                "next_phase": "modeling",
                "reasoning": "User context suggests modeling is needed",
                "requires_iteration": False,
                "priority": "medium"
            }
        else:
            return {
                "next_phase": "deployment_preparation",
                "reasoning": "Ready for final recommendations",
                "requires_iteration": False,
                "priority": "high"
            }
    
    def _prepare_final_phase(self, state: AgentState) -> Dict[str, Any]:
        """Prepare for the final deployment preparation phase."""
        return {
            "success": True,
            "findings": {"coordinator_decision": "Final phase due to time/iteration limits"},
            "notebook_cells": [],
            "next_phase": "deployment_preparation",
            "message": f"Moving to final phase - Time remaining: {state.time_remaining:.0f}s, Iterations: {state.iteration_count}/{state.max_iterations}",
            "requires_iteration": False
        }