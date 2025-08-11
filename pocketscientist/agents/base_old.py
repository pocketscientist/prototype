"""
Base agent class for CRISP-DM workflow.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from ..llm.base import BaseLLMProvider
from ..execution import CellExecutor, ExecutionResult


@dataclass
class AgentState:
    """Shared state between agents."""
    dataset_path: str
    user_context: str
    findings: Dict[str, Any] = field(default_factory=dict)
    notebook_cells: List[Dict[str, Any]] = field(default_factory=list)
    current_phase: str = "business_understanding"
    phase_history: List[str] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    max_time: int = 1800  # 30 minutes default
    time_remaining: float = field(default=0.0)
    iteration_count: int = 0
    max_iterations: int = 20
    completed_phases: set = field(default_factory=set)
    errors: List[str] = field(default_factory=list)
    execution_results: Dict[str, Any] = field(default_factory=dict)  # Store execution outcomes
    namespace_summary: Dict[str, str] = field(default_factory=dict)  # Current variable state
    agent_learnings: Dict[str, Any] = field(default_factory=dict)  # What agents have learned
    useful_patterns: List[str] = field(default_factory=list)  # Successful code patterns
    failed_operations: List[Dict[str, Any]] = field(default_factory=list)  # Failed attempts
    
    def update_time_remaining(self) -> None:
        """Update the remaining time."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.time_remaining = max(0, self.max_time - elapsed)
    
    def has_time_remaining(self) -> bool:
        """Check if there's time remaining."""
        self.update_time_remaining()
        return self.time_remaining > 0
    
    def can_continue_iteration(self) -> bool:
        """Check if we can continue with more iterations."""
        return (
            self.iteration_count < self.max_iterations 
            and self.has_time_remaining()
        )


class BaseAgent(ABC):
    """Base class for all CRISP-DM agents."""
    
    def __init__(
        self,
        name: str,
        llm_provider: BaseLLMProvider,
        phase_name: str,
        executor: Optional[CellExecutor] = None
    ) -> None:
        self.name = name
        self.llm_provider = llm_provider
        self.phase_name = phase_name
        self.executor = executor
    
    @abstractmethod
    def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        Execute the agent's task.
        
        Args:
            state: Current shared state
            
        Returns:
            Dictionary containing:
            - success: bool - Whether execution was successful
            - findings: Dict - New findings to add to state
            - notebook_cells: List - New notebook cells to add
            - next_phase: Optional[str] - Recommended next phase
            - message: str - Status message
            - requires_iteration: bool - Whether this phase needs more work
        """
        pass
    
    def _get_system_prompt(self, state: Optional[AgentState] = None) -> str:
        """Get the system prompt for this agent."""
        
        base_prompt = f"""You are a {self.name} agent in a CRISP-DM data science workflow.
Your role is to {self.phase_name} as part of a comprehensive data analysis process.

You have access to various Python libraries for data science:
- pandas for data manipulation
- numpy for numerical operations
- matplotlib and seaborn for visualization
- scikit-learn for machine learning
- plotly for interactive plots

When generating code:
1. Create executable Python code that can be run in a Jupyter notebook
2. Include proper imports
3. Add comments explaining your approach
4. Handle errors gracefully
5. Generate meaningful outputs (prints, plots, summaries)
6. Save any important plots or results

Always think step by step and explain your reasoning."""

        # Add execution context if available
        if state:
            context_parts = [base_prompt]
            
            # Add current variable state
            if state.namespace_summary:
                context_parts.append("\nCurrent variables available:")
                for var_name, var_info in state.namespace_summary.items():
                    context_parts.append(f"- {var_name}: {var_info}")
            
            # Add recent execution results with more detail
            if state.execution_results:
                context_parts.append("\nRecent execution results:")
                recent_results = list(state.execution_results.items())[-5:]  # Last 5 results
                for key, result in recent_results:
                    status = "✓" if result["success"] else "✗"
                    context_parts.append(f"- {key}: {status}")
                    if result.get("output") and result["success"]:
                        # Show successful output to inform next steps
                        output_preview = result["output"][:200].strip()
                        if output_preview:
                            context_parts.append(f"  Output: {output_preview}...")
                    if result.get("error"):
                        context_parts.append(f"  Error: {result['error'][:100]}...")
                    if result.get("variables_created"):
                        context_parts.append(f"  Created: {', '.join(result['variables_created'])}")
            
            # Add agent learnings
            if hasattr(state, 'agent_learnings') and state.agent_learnings:
                context_parts.append("\nLearnings from previous attempts:")
                for key, learning in list(state.agent_learnings.items())[-3:]:
                    context_parts.append(f"- {key}: {learning}")
            
            # Add useful patterns discovered
            if hasattr(state, 'useful_patterns') and state.useful_patterns:
                context_parts.append("\nUseful patterns discovered:")
                for pattern in state.useful_patterns[-5:]:
                    context_parts.append(f"- {pattern}")
            
            # Add dataset context
            context_parts.append(f"\nDataset: {state.dataset_path}")
            context_parts.append(f"Analysis goal: {state.user_context}")
            
            return "\n".join(context_parts)
        
        return base_prompt
    
    def _generate_code_cell(
        self,
        prompt: str,
        cell_type: str = "code",
        metadata: Optional[Dict[str, Any]] = None,
        state: Optional[AgentState] = None,
        execute: bool = True
    ) -> Dict[str, Any]:
        """Generate a notebook cell using the LLM and optionally execute it."""
        system_prompt = self._get_system_prompt(state)
        
        if cell_type == "code":
            full_prompt = f"""
{prompt}

Generate Python code that:
1. Is executable in a Jupyter notebook environment
2. Includes necessary imports
3. Has clear comments
4. Handles potential errors
5. Produces meaningful output

Return only the Python code, no markdown formatting or explanation.
"""
        else:
            full_prompt = f"""
{prompt}

Generate markdown content that explains the current step in the data science process.
Be clear, concise, and informative.

Return only the markdown content, no code blocks.
"""
        
        try:
            content = self.llm_provider.generate(
                prompt=full_prompt,
                system_prompt=system_prompt,
                temperature=0.3
            )
            
            cell = {
                "cell_type": cell_type,
                "metadata": metadata or {},
                "source": content.strip().split('\n'),
                "execution_count": None,
                "outputs": []
            }
            
            # Execute code cell if requested and executor is available
            if cell_type == "code" and execute and self.executor and state:
                # First execution attempt
                execution_result, final_code = self.executor.execute_code(
                    code=content.strip(),
                    max_retries=5
                )
                
                # If execution failed, try to regenerate with error context
                if not execution_result.success and execution_result.error:
                    # Analyze the error and try to fix
                    error_context = f"""
The previous code failed with error:
{execution_result.error}

Code that failed:
{final_code[:500]}

Please generate corrected code that addresses this error.
Available variables: {', '.join(state.namespace_summary.keys()) if state.namespace_summary else 'None yet'}
"""
                    
                    # Regenerate with error context
                    corrected_content = self.llm_provider.generate(
                        prompt=full_prompt + "\n\n" + error_context,
                        system_prompt=system_prompt,
                        temperature=0.3
                    )
                    
                    # Try executing the corrected code
                    execution_result_2, final_code_2 = self.executor.execute_code(
                        code=corrected_content.strip(),
                        max_retries=3
                    )
                    
                    # Use the corrected version if it succeeded
                    if execution_result_2.success:
                        execution_result = execution_result_2
                        final_code = final_code_2
                        
                        # Learn from this correction
                        if not hasattr(state, 'agent_learnings'):
                            state.agent_learnings = {}
                        state.agent_learnings[f"{self.phase_name}_correction"] = {
                            "original_error": execution_result.error[:100],
                            "correction_succeeded": True
                        }
                
                # Update cell with execution results
                cell["source"] = final_code.split('\n')
                cell["execution_count"] = self.executor.execution_count
                
                # Create output cells
                outputs = []
                
                if execution_result.output:
                    outputs.append({
                        "output_type": "stream",
                        "name": "stdout",
                        "text": execution_result.output.split('\n')
                    })
                
                if execution_result.error:
                    outputs.append({
                        "output_type": "error",
                        "ename": "ExecutionError",
                        "evalue": execution_result.error,
                        "traceback": execution_result.error.split('\n')
                    })
                
                # Add plot outputs
                for plot_b64 in execution_result.plots:
                    outputs.append({
                        "output_type": "display_data",
                        "data": {
                            "image/png": plot_b64
                        },
                        "metadata": {}
                    })
                
                cell["outputs"] = outputs
                
                # Store execution results in state for agent to observe
                if state:
                    execution_key = f"{self.phase_name}_{len(state.notebook_cells)}"
                    state.execution_results[execution_key] = {
                        "success": execution_result.success,
                        "output": execution_result.output,
                        "error": execution_result.error,
                        "variables_created": execution_result.variables_created,
                        "execution_time": execution_result.execution_time,
                        "plots_count": len(execution_result.plots)
                    }
                    
                    # Update namespace summary
                    state.namespace_summary = self.executor.get_namespace_summary()
                    
                    # Learn from successful execution
                    self._learn_from_success(state, final_code, execution_result)
            
            return cell
            
        except Exception as e:
            # Fallback cell on error
            error_content = f"# Error generating {cell_type} cell: {str(e)}"
            return {
                "cell_type": "markdown",
                "metadata": {},
                "source": [error_content],
                "execution_count": None,
                "outputs": []
            }
    
    def _learn_from_success(self, state: Any, code: str, result: Any) -> None:
        """Learn from successful code execution."""
        
        # Initialize useful patterns if not exists
        if not hasattr(state, 'useful_patterns'):
            state.useful_patterns = []
        
        # Learn patterns from successful code
        if "pd.read_csv" in code and result.success:
            if "Data successfully loaded with pd.read_csv" not in state.useful_patterns:
                state.useful_patterns.append("Data successfully loaded with pd.read_csv")
        
        if "df.shape" in code and result.output:
            if "Dataset shape retrieved" not in state.useful_patterns:
                state.useful_patterns.append("Dataset shape retrieved")
        
        if "plt.show()" in code or "plt.savefig" in code:
            if "Visualization created successfully" not in state.useful_patterns:
                state.useful_patterns.append("Visualization created successfully")
        
        if "train_test_split" in code and result.success:
            if "Data split for training/testing" not in state.useful_patterns:
                state.useful_patterns.append("Data split for training/testing")
        
        # Store specific insights from output
        if result.output and "shape:" in result.output.lower():
            # Extract shape information
            import re
            shape_match = re.search(r'shape[:\s]+\((\d+),\s*(\d+)\)', result.output.lower())
            if shape_match:
                rows, cols = shape_match.groups()
                state.useful_patterns.append(f"Dataset has {rows} rows and {cols} columns")
    
    def _add_markdown_cell(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a markdown cell with the given content."""
        return {
            "cell_type": "markdown",
            "metadata": metadata or {},
            "source": content.strip().split('\n'),
            "execution_count": None,
            "outputs": []
        }