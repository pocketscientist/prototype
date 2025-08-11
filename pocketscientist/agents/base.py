"""
Base agent class for CRISP-DM workflow.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from ..llm.base import BaseLLMProvider


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
        phase_name: str
    ) -> None:
        self.name = name
        self.llm_provider = llm_provider
        self.phase_name = phase_name
    
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
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        return f"""You are a {self.name} agent in a CRISP-DM data science workflow.
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
    
    def _generate_code_cell(
        self,
        prompt: str,
        cell_type: str = "code",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate a notebook cell using the LLM."""
        system_prompt = self._get_system_prompt()
        
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