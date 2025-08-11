"""
Main orchestrator for PocketScientist data science workflow.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from .llm.factory import create_llm_provider
from .agents import (
    BusinessUnderstandingAgent,
    DataUnderstandingAgent,
    DataPreparationAgent, 
    ModelingAgent,
    EvaluationAgent,
    DeploymentPreparationAgent,
    CoordinatorAgent
)
from .agents.base import AgentState
from .notebook_builder import NotebookBuilder
from .report_generator import ReportGenerator
from .utils.safety import SafetyMonitor
from .utils.validation import validate_dataset


class DataScienceOrchestrator:
    """Main orchestrator for CRISP-DM workflow."""
    
    def __init__(
        self,
        dataset_path: str,
        context: str,
        output_dir: Path,
        max_time: int = 1800,
        model_endpoint: str = "http://localhost:11434",
        model_name: str = "llama3.1", 
        api_key: Optional[str] = None,
        verbose: bool = False
    ):
        self.dataset_path = dataset_path
        self.context = context
        self.output_dir = output_dir
        self.max_time = max_time
        self.verbose = verbose
        
        # Initialize LLM provider
        self.llm_provider = create_llm_provider(
            endpoint=model_endpoint,
            model_name=model_name,
            api_key=api_key
        )
        
        # Initialize agents
        self.agents = {
            "business_understanding": BusinessUnderstandingAgent(self.llm_provider),
            "data_understanding": DataUnderstandingAgent(self.llm_provider),
            "data_preparation": DataPreparationAgent(self.llm_provider),
            "modeling": ModelingAgent(self.llm_provider),
            "evaluation": EvaluationAgent(self.llm_provider),
            "deployment_preparation": DeploymentPreparationAgent(self.llm_provider),
            "coordinator": CoordinatorAgent(self.llm_provider)
        }
        
        # Initialize builders
        self.notebook_builder = NotebookBuilder()
        self.report_generator = ReportGenerator()
        
        # Initialize safety monitor
        self.safety_monitor = SafetyMonitor(
            max_time=max_time,
            max_iterations=20,
            max_phase_repeats=3
        )
        
        # Initialize state
        self.state = AgentState(
            dataset_path=dataset_path,
            user_context=context,
            start_time=datetime.now(),
            max_time=max_time
        )
    
    def run(self) -> Dict[str, Any]:
        """Run the complete CRISP-DM workflow."""
        
        try:
            # Validate dataset first
            dataset_validation = validate_dataset(self.dataset_path)
            if not dataset_validation["valid"]:
                error_msg = f"Dataset validation failed: {'; '.join(dataset_validation['errors'])}"
                return {
                    "success": False,
                    "error": error_msg,
                    "notebook_path": None,
                    "report_path": None
                }
            
            if dataset_validation["warnings"] and self.verbose:
                for warning in dataset_validation["warnings"]:
                    print(f"⚠️ Dataset warning: {warning}")
            
            # Validate LLM connection
            if not self._validate_llm_connection():
                return {
                    "success": False,
                    "error": "Could not connect to LLM provider",
                    "notebook_path": None,
                    "report_path": None
                }
            
            if self.verbose:
                print(f"🚀 Starting CRISP-DM analysis...")
                print(f"📊 Dataset: {self.dataset_path}")
                print(f"❓ Context: {self.context}")
                print(f"⏰ Max time: {self.max_time} seconds")
            
            # Execute workflow
            workflow_result = self._execute_workflow()
            
            if not workflow_result["success"]:
                return workflow_result
            
            # Build final outputs
            notebook_path = self._build_notebook()
            report_path = self._generate_report()
            
            # Get execution statistics
            execution_stats = self.safety_monitor.get_execution_stats()
            
            return {
                "success": True,
                "error": None,
                "notebook_path": notebook_path,
                "report_path": report_path,
                "execution_time": (datetime.now() - self.state.start_time).total_seconds(),
                "iterations": self.state.iteration_count,
                "phases_completed": list(self.state.completed_phases),
                "final_findings": self.state.findings,
                "execution_stats": execution_stats,
                "safety_warnings": len(self.state.errors)
            }
            
        except Exception as e:
            if self.verbose:
                import traceback
                traceback.print_exc()
                
            return {
                "success": False, 
                "error": str(e),
                "notebook_path": None,
                "report_path": None
            }
    
    def _validate_llm_connection(self) -> bool:
        """Validate connection to LLM provider."""
        try:
            validation = self.llm_provider.validate_connection()
            if self.verbose and not validation["success"]:
                print(f"❌ LLM connection failed: {validation['error']}")
            return validation["success"]
        except Exception as e:
            if self.verbose:
                print(f"❌ LLM validation error: {str(e)}")
            return False
    
    def _execute_workflow(self) -> Dict[str, Any]:
        """Execute the main CRISP-DM workflow."""
        
        # Start with business understanding
        current_phase = "business_understanding"
        
        while current_phase and self.state.can_continue_iteration():
            self.state.update_time_remaining()
            self.state.iteration_count += 1
            
            # Check safety before each phase
            safety_check = self.safety_monitor.check_safety(
                current_phase,
                self.state.iteration_count,
                self.state.phase_history
            )
            
            if not safety_check["safe_to_continue"]:
                if self.verbose:
                    print(f"🛑 Safety check failed: {safety_check['reason']}")
                self.state.errors.append(f"Safety termination: {safety_check['reason']}")
                break
            
            # Log safety warnings
            for warning in safety_check.get("warnings", []):
                if self.verbose:
                    print(f"⚠️ Safety warning: {warning}")
                self.state.errors.append(f"Warning: {warning}")
            
            if self.verbose:
                print(f"\n🔄 Iteration {self.state.iteration_count}: Executing {current_phase}")
                print(f"⏰ Time remaining: {self.state.time_remaining:.0f}s")
            
            # Record phase start
            self.safety_monitor.record_phase_start(current_phase)
            
            # Execute current phase
            try:
                agent = self.agents[current_phase]
                result = agent.execute(self.state)
                
                # Record phase end
                self.safety_monitor.record_phase_end(current_phase, result["success"])
                
                if not result["success"]:
                    self.state.errors.append(f"{current_phase}: {result['message']}")
                    if self.verbose:
                        print(f"⚠️ Phase {current_phase} failed: {result['message']}")
                else:
                    if self.verbose:
                        print(f"✅ Phase {current_phase} completed: {result['message']}")
                
                # Update state
                if result.get("findings"):
                    self.state.findings[current_phase] = result["findings"]
                
                if result.get("notebook_cells"):
                    self.state.notebook_cells.extend(result["notebook_cells"])
                
                self.state.completed_phases.add(current_phase)
                self.state.phase_history.append(current_phase)
                self.state.current_phase = current_phase
                
                # Determine next phase
                if result.get("next_phase"):
                    if result["next_phase"] == "coordinator":
                        # Use coordinator to decide
                        coordinator_result = self.agents["coordinator"].execute(self.state)
                        if coordinator_result.get("next_phase"):
                            current_phase = coordinator_result["next_phase"]
                        else:
                            current_phase = None
                    else:
                        current_phase = result["next_phase"]
                else:
                    current_phase = None
                
                # Add a small delay to prevent overwhelming the LLM
                time.sleep(1)
                
            except Exception as e:
                error_msg = f"Error in {current_phase}: {str(e)}"
                self.state.errors.append(error_msg)
                
                # Record phase end as failed
                self.safety_monitor.record_phase_end(current_phase, success=False)
                
                if self.verbose:
                    print(f"❌ {error_msg}")
                
                # Try to continue with next logical phase
                current_phase = self._get_fallback_next_phase(current_phase)
        
        if self.verbose:
            print(f"\n🏁 Workflow completed after {self.state.iteration_count} iterations")
            print(f"📋 Phases completed: {', '.join(self.state.completed_phases)}")
        
        return {"success": True}
    
    def _get_fallback_next_phase(self, failed_phase: str) -> Optional[str]:
        """Get fallback next phase when current phase fails."""
        
        phase_sequence = [
            "business_understanding",
            "data_understanding", 
            "data_preparation",
            "modeling",
            "evaluation",
            "deployment_preparation"
        ]
        
        try:
            current_index = phase_sequence.index(failed_phase)
            if current_index < len(phase_sequence) - 1:
                return phase_sequence[current_index + 1]
        except ValueError:
            pass
        
        return "deployment_preparation"  # Always end with this
    
    def _build_notebook(self) -> str:
        """Build the final Jupyter notebook."""
        notebook_path = self.output_dir / "analysis.ipynb"
        
        try:
            self.notebook_builder.build_notebook(
                cells=self.state.notebook_cells,
                output_path=notebook_path,
                title=f"PocketScientist Analysis: {Path(self.dataset_path).stem}",
                context=self.context
            )
            
            if self.verbose:
                print(f"📓 Notebook saved: {notebook_path}")
            
            return str(notebook_path)
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Notebook building failed: {str(e)}")
            return ""
    
    def _generate_report(self) -> str:
        """Generate the final HTML report."""
        report_path = self.output_dir / "report.html"
        
        try:
            self.report_generator.generate_report(
                state=self.state,
                output_path=report_path,
                notebook_path=self.output_dir / "analysis.ipynb"
            )
            
            if self.verbose:
                print(f"📋 Report saved: {report_path}")
            
            return str(report_path)
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Report generation failed: {str(e)}")
            return ""