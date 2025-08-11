"""
Base agent class for CRISP-DM workflow with enhanced execution observation and learning.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
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
    """Enhanced base class with execution observation and learning."""
    
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
        self.execution_history = []  # Track all executions for learning
    
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
    
    def _generate_code_cell(
        self,
        prompt: str,
        state: Optional[AgentState] = None,
        metadata: Optional[Dict[str, Any]] = None,
        max_attempts: int = 3
    ) -> Dict[str, Any]:
        """
        Generate code, execute it, observe results, and learn from failures.
        
        This method will:
        1. Generate initial code
        2. Execute and observe results
        3. If failed, analyze error and regenerate with corrections
        4. Learn from the process and update context
        """
        
        attempt = 0
        final_cell = None
        all_attempts = []
        
        while attempt < max_attempts:
            attempt += 1
            
            # Build enhanced prompt with execution context
            enhanced_prompt = self._build_enhanced_prompt(prompt, state, all_attempts)
            
            # Generate code
            generated_code = self._generate_code(enhanced_prompt, state)
            
            if not self.executor or not state:
                # No executor, just return the generated code
                return self._create_cell_dict(generated_code, "code", metadata)
            
            # Execute the code
            execution_result, final_code = self.executor.execute_code(
                code=generated_code,
                max_retries=2  # Quick retries for syntax errors
            )
            
            # Record this attempt
            attempt_record = {
                "attempt": attempt,
                "code": generated_code,
                "final_code": final_code,
                "success": execution_result.success,
                "output": execution_result.output,
                "error": execution_result.error,
                "variables_created": execution_result.variables_created
            }
            all_attempts.append(attempt_record)
            
            # If successful, we're done
            if execution_result.success:
                final_cell = self._create_executed_cell(
                    final_code, 
                    execution_result, 
                    metadata
                )
                
                # Learn from success
                self._learn_from_execution(state, attempt_record, True)
                break
            
            # If failed and we have more attempts, analyze and retry
            if attempt < max_attempts:
                # Analyze the error and get suggestions for fixing
                error_analysis = self._analyze_execution_error(
                    execution_result.error,
                    final_code,
                    state
                )
                
                # Add error analysis to context for next attempt
                all_attempts[-1]["error_analysis"] = error_analysis
                
                # Learn from failure
                self._learn_from_execution(state, attempt_record, False)
            else:
                # Final attempt failed, create cell with error
                final_cell = self._create_executed_cell(
                    final_code,
                    execution_result,
                    metadata
                )
        
        # Store complete execution history
        self.execution_history.append({
            "prompt": prompt,
            "attempts": all_attempts,
            "final_success": all_attempts[-1]["success"] if all_attempts else False
        })
        
        # Update state with what we learned
        self._update_state_with_learnings(state, all_attempts)
        
        return final_cell
    
    def _build_enhanced_prompt(
        self,
        base_prompt: str,
        state: AgentState,
        previous_attempts: List[Dict[str, Any]]
    ) -> str:
        """Build an enhanced prompt with execution context and previous attempts."""
        
        parts = [base_prompt]
        
        # Add context from previous attempts if any failed
        if previous_attempts:
            failed_attempts = [a for a in previous_attempts if not a["success"]]
            if failed_attempts:
                parts.append("\n\nPREVIOUS ATTEMPTS AND ERRORS:")
                for attempt in failed_attempts:
                    parts.append(f"\nAttempt {attempt['attempt']} failed with error:")
                    parts.append(f"Code: {attempt['code'][:200]}...")
                    parts.append(f"Error: {attempt['error'][:200]}...")
                    if attempt.get("error_analysis"):
                        parts.append(f"Analysis: {attempt['error_analysis']}")
                
                parts.append("\nPlease generate corrected code that addresses these issues.")
        
        # Add current execution context
        if state and state.namespace_summary:
            parts.append("\n\nCURRENT VARIABLES IN MEMORY:")
            for var_name, var_info in list(state.namespace_summary.items())[:10]:
                parts.append(f"- {var_name}: {var_info}")
        
        # Add recent successful operations
        if state and state.execution_results:
            successful = [r for r in state.execution_results.values() if r.get("success")]
            if successful:
                parts.append("\n\nRECENT SUCCESSFUL OPERATIONS:")
                for result in successful[-3:]:
                    if result.get("output"):
                        parts.append(f"- Output: {result['output'][:100]}...")
        
        return "\n".join(parts)
    
    def _analyze_execution_error(
        self,
        error: str,
        code: str,
        state: AgentState
    ) -> str:
        """Analyze execution error and provide fix suggestions."""
        
        prompt = f"""
Analyze this Python code execution error and suggest a fix:

ERROR:
{error}

CODE THAT FAILED:
{code}

AVAILABLE VARIABLES:
{', '.join(state.namespace_summary.keys()) if state and state.namespace_summary else 'Unknown'}

Provide a brief analysis of what went wrong and how to fix it.
Focus on:
1. What caused the error
2. What needs to be changed
3. Specific code corrections needed
"""
        
        try:
            analysis = self.llm_provider.generate(
                prompt=prompt,
                system_prompt="You are a Python debugging expert. Provide concise, actionable error analysis.",
                temperature=0.3
            )
            return analysis
        except:
            # Fallback analysis
            if "NameError" in error:
                return "Variable not defined. Check variable names and ensure proper imports."
            elif "AttributeError" in error:
                return "Attribute doesn't exist. Check object types and available methods."
            elif "KeyError" in error:
                return "Key not found. Check dictionary/dataframe columns."
            elif "FileNotFoundError" in error:
                return "File not found. Check file path and working directory."
            else:
                return "Execution error. Check syntax and logic."
    
    def _learn_from_execution(
        self,
        state: AgentState,
        attempt_record: Dict[str, Any],
        success: bool
    ) -> None:
        """Learn from execution attempt and update agent knowledge."""
        
        # Store what worked or didn't work
        learning_key = f"{self.phase_name}_learning_{len(self.execution_history)}"
        
        state.agent_learnings[learning_key] = {
            "success": success,
            "code_pattern": attempt_record["code"][:100],
            "outcome": "SUCCESS" if success else f"FAILED: {attempt_record['error'][:100]}",
            "variables_created": attempt_record.get("variables_created", [])
        }
        
        # If successful, remember useful patterns
        if success and attempt_record.get("variables_created"):
            # Extract useful code patterns (like how data was loaded, etc.)
            code = attempt_record["code"]
            if "pd.read_csv" in code:
                if "Data successfully loaded with pd.read_csv" not in state.useful_patterns:
                    state.useful_patterns.append("Data successfully loaded with pd.read_csv")
            if "plt." in code:
                if "Visualization created with matplotlib" not in state.useful_patterns:
                    state.useful_patterns.append("Visualization created with matplotlib")
            if "train_test_split" in code:
                if "Data split for training/testing" not in state.useful_patterns:
                    state.useful_patterns.append("Data split for training/testing")
            
            # Extract specific insights from output
            if attempt_record.get("output"):
                output = attempt_record["output"].lower()
                if "shape:" in output or "shape (" in output:
                    import re
                    shape_match = re.search(r'(\d+)[,\s]+(\d+)', output)
                    if shape_match:
                        rows, cols = shape_match.groups()
                        pattern = f"Dataset has {rows} rows and {cols} columns"
                        if pattern not in state.useful_patterns:
                            state.useful_patterns.append(pattern)
    
    def _update_state_with_learnings(
        self,
        state: AgentState,
        attempts: List[Dict[str, Any]]
    ) -> None:
        """Update state with learnings from all attempts."""
        
        # Find the successful attempt if any
        successful = [a for a in attempts if a["success"]]
        
        if successful:
            last_success = successful[-1]
            # Update execution results with success
            exec_key = f"{self.phase_name}_{len(state.notebook_cells)}"
            state.execution_results[exec_key] = {
                "success": True,
                "output": last_success["output"],
                "variables_created": last_success["variables_created"],
                "attempts_needed": len(attempts)
            }
            
            # Update namespace if we have executor
            if self.executor:
                state.namespace_summary = self.executor.get_namespace_summary()
        else:
            # All attempts failed - record this for coordinator to handle
            state.failed_operations.append({
                "phase": self.phase_name,
                "attempts": len(attempts),
                "last_error": attempts[-1]["error"] if attempts else "Unknown"
            })
    
    def _generate_code(self, prompt: str, state: AgentState) -> str:
        """Generate code using LLM with full context."""
        
        full_prompt = f"""
{prompt}

Generate Python code that:
1. Is executable in a Jupyter notebook environment
2. Includes necessary imports
3. Has clear comments
4. Handles potential errors gracefully
5. Produces meaningful output (prints, plots, summaries)
6. Uses variables that are already available when possible

Return only the Python code, no markdown formatting or explanation.
"""
        
        system_prompt = self._get_system_prompt(state)
        
        try:
            code = self.llm_provider.generate(
                prompt=full_prompt,
                system_prompt=system_prompt,
                temperature=0.3
            )
            # Clean up code formatting
            code = code.strip()
            # Remove markdown code blocks if present
            if code.startswith("```"):
                lines = code.split('\n')
                code = '\n'.join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            return code
        except Exception as e:
            # Fallback code
            return f"# Error generating code: {str(e)}\nprint('Code generation failed')"
    
    def _create_cell_dict(
        self,
        source: str,
        cell_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a basic cell dictionary."""
        return {
            "cell_type": cell_type,
            "metadata": metadata or {},
            "source": source.split('\n'),
            "execution_count": None,
            "outputs": []
        }
    
    def _create_executed_cell(
        self,
        code: str,
        execution_result: ExecutionResult,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a cell with execution results."""
        
        cell = self._create_cell_dict(code, "code", metadata)
        
        if self.executor:
            cell["execution_count"] = self.executor.execution_count
        
        # Add outputs
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
        
        # Add plots
        for plot_b64 in execution_result.plots:
            outputs.append({
                "output_type": "display_data",
                "data": {
                    "image/png": plot_b64
                },
                "metadata": {}
            })
        
        cell["outputs"] = outputs
        
        return cell
    
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
    
    def _get_system_prompt(self, state: Optional[AgentState] = None) -> str:
        """Get enhanced system prompt with execution context."""
        
        base_prompt = f"""You are a {self.name} agent in a CRISP-DM data science workflow.
Your role is to {self.phase_name} as part of a comprehensive data analysis process.

You have access to various Python libraries for data science:
- pandas for data manipulation
- numpy for numerical operations  
- matplotlib and seaborn for visualization
- scikit-learn for machine learning
- plotly for interactive plots

When generating code:
1. Use variables that already exist when possible
2. Create executable Python code for Jupyter notebooks
3. Include proper imports
4. Add helpful comments
5. Handle errors gracefully
6. Generate meaningful outputs
7. Learn from previous execution results

Always think step by step and explain your reasoning through code comments."""

        if state:
            context_parts = [base_prompt]
            
            # Add learned patterns
            if state.useful_patterns:
                context_parts.append("\nLEARNED PATTERNS FROM PREVIOUS EXECUTIONS:")
                for pattern in state.useful_patterns[-5:]:
                    context_parts.append(f"- {pattern}")
            
            # Add agent learnings
            if state.agent_learnings:
                recent_learnings = list(state.agent_learnings.items())[-3:]
                context_parts.append("\nRECENT EXECUTION LEARNINGS:")
                for key, learning in recent_learnings:
                    status = "✓" if learning["success"] else "✗"
                    context_parts.append(f"- {key}: {status} {learning['outcome']}")
            
            # Add current variables
            if state.namespace_summary:
                context_parts.append("\nCURRENT VARIABLES AVAILABLE:")
                for var_name, var_info in list(state.namespace_summary.items())[:10]:
                    context_parts.append(f"- {var_name}: {var_info}")
            
            # Add recent execution results
            if state.execution_results:
                context_parts.append("\nRECENT EXECUTION RESULTS:")
                recent = list(state.execution_results.items())[-3:]
                for key, result in recent:
                    if result.get("success") and result.get("output"):
                        output_preview = result["output"][:100].strip()
                        if output_preview:
                            context_parts.append(f"- {key}: {output_preview}...")
            
            # Add dataset info
            context_parts.append(f"\nDataset: {state.dataset_path}")
            context_parts.append(f"Analysis goal: {state.user_context}")
            
            return "\n".join(context_parts)
        
        return base_prompt