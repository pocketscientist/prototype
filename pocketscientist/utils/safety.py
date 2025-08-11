"""
Safety monitoring and loop prevention utilities.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging


class SafetyMonitor:
    """Monitor workflow execution for safety and prevent infinite loops."""
    
    def __init__(
        self,
        max_time: int = 1800,
        max_iterations: int = 20,
        max_phase_repeats: int = 3,
        min_phase_time: float = 5.0
    ):
        self.max_time = max_time
        self.max_iterations = max_iterations
        self.max_phase_repeats = max_phase_repeats
        self.min_phase_time = min_phase_time
        
        self.start_time = datetime.now()
        self.phase_counts: Dict[str, int] = {}
        self.phase_timings: List[Dict[str, Any]] = []
        self.warnings: List[str] = []
        
        self.logger = logging.getLogger(__name__)
    
    def check_safety(
        self,
        current_phase: str,
        iteration_count: int,
        phase_history: List[str]
    ) -> Dict[str, Any]:
        """
        Check if execution should continue safely.
        
        Returns:
            Dictionary with safety status and recommendations
        """
        
        safety_result = {
            "safe_to_continue": True,
            "warnings": [],
            "should_terminate": False,
            "reason": None
        }
        
        # Check time limit
        elapsed = (datetime.now() - self.start_time).total_seconds()
        time_remaining = self.max_time - elapsed
        
        if elapsed >= self.max_time:
            safety_result.update({
                "safe_to_continue": False,
                "should_terminate": True,
                "reason": "Maximum execution time exceeded"
            })
            return safety_result
        
        # Check iteration limit
        if iteration_count >= self.max_iterations:
            safety_result.update({
                "safe_to_continue": False,
                "should_terminate": True,
                "reason": "Maximum iteration count exceeded"
            })
            return safety_result
        
        # Check for excessive phase repetition
        phase_repeat_count = self._count_recent_phase_repeats(current_phase, phase_history)
        if phase_repeat_count >= self.max_phase_repeats:
            safety_result.update({
                "safe_to_continue": False,
                "should_terminate": True,
                "reason": f"Phase '{current_phase}' repeated {phase_repeat_count} times - possible infinite loop"
            })
            return safety_result
        
        # Check for rapid phase cycling
        if self._detect_rapid_cycling(phase_history):
            safety_result["warnings"].append(
                "Rapid phase cycling detected - workflow may be stuck in a loop"
            )
        
        # Check for time warnings
        if time_remaining < 300:  # 5 minutes warning
            safety_result["warnings"].append(
                f"Warning: Only {time_remaining:.0f} seconds remaining"
            )
        
        # Check for excessive back-and-forth
        if self._detect_excessive_backtracking(phase_history):
            safety_result["warnings"].append(
                "Excessive backtracking detected - consider simplifying analysis"
            )
        
        return safety_result
    
    def record_phase_start(self, phase: str) -> None:
        """Record the start of a phase."""
        self.phase_counts[phase] = self.phase_counts.get(phase, 0) + 1
        self.phase_timings.append({
            "phase": phase,
            "start_time": datetime.now(),
            "end_time": None,
            "duration": None
        })
    
    def record_phase_end(self, phase: str, success: bool = True) -> None:
        """Record the end of a phase."""
        if self.phase_timings and self.phase_timings[-1]["phase"] == phase:
            end_time = datetime.now()
            self.phase_timings[-1]["end_time"] = end_time
            self.phase_timings[-1]["duration"] = (
                end_time - self.phase_timings[-1]["start_time"]
            ).total_seconds()
            self.phase_timings[-1]["success"] = success
    
    def _count_recent_phase_repeats(self, phase: str, history: List[str]) -> int:
        """Count recent repetitions of a phase."""
        if len(history) < 3:
            return 0
        
        # Look at the last few phases to detect immediate repetition
        recent_history = history[-5:]  # Last 5 phases
        return recent_history.count(phase)
    
    def _detect_rapid_cycling(self, history: List[str]) -> bool:
        """Detect if phases are cycling rapidly."""
        if len(history) < 6:
            return False
        
        # Check if we're cycling through the same 2-3 phases repeatedly
        recent = history[-6:]
        unique_recent = set(recent)
        
        # If we have very few unique phases but many total phases, we might be cycling
        if len(unique_recent) <= 2 and len(recent) >= 4:
            return True
        
        return False
    
    def _detect_excessive_backtracking(self, history: List[str]) -> bool:
        """Detect excessive backtracking to earlier phases."""
        if len(history) < 4:
            return False
        
        # Define phase order
        phase_order = [
            "business_understanding",
            "data_understanding", 
            "data_preparation",
            "modeling",
            "evaluation",
            "deployment_preparation"
        ]
        
        # Count backward transitions
        backward_count = 0
        for i in range(1, len(history)):
            prev_phase = history[i-1]
            curr_phase = history[i]
            
            if prev_phase in phase_order and curr_phase in phase_order:
                prev_index = phase_order.index(prev_phase)
                curr_index = phase_order.index(curr_phase)
                
                if curr_index < prev_index:
                    backward_count += 1
        
        # If more than 50% of transitions are backward, that's excessive
        total_transitions = len(history) - 1
        if total_transitions > 0:
            backward_ratio = backward_count / total_transitions
            return backward_ratio > 0.5
        
        return False
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "total_time": elapsed,
            "time_remaining": max(0, self.max_time - elapsed),
            "phase_counts": self.phase_counts.copy(),
            "phase_timings": self.phase_timings.copy(),
            "warnings": self.warnings.copy(),
            "avg_phase_time": self._calculate_avg_phase_time(),
            "efficiency_score": self._calculate_efficiency_score()
        }
    
    def _calculate_avg_phase_time(self) -> float:
        """Calculate average phase execution time."""
        completed_phases = [p for p in self.phase_timings if p.get("duration")]
        if not completed_phases:
            return 0.0
        
        total_time = sum(p["duration"] for p in completed_phases)
        return total_time / len(completed_phases)
    
    def _calculate_efficiency_score(self) -> float:
        """Calculate workflow efficiency score (0-100)."""
        if not self.phase_timings:
            return 0.0
        
        # Base score
        score = 100.0
        
        # Penalize for excessive repetitions
        total_phases = sum(self.phase_counts.values())
        unique_phases = len(self.phase_counts)
        if unique_phases > 0:
            repetition_ratio = total_phases / unique_phases
            if repetition_ratio > 2:
                score -= (repetition_ratio - 2) * 10
        
        # Penalize for warnings
        score -= len(self.warnings) * 5
        
        # Penalize for failures
        failed_phases = len([p for p in self.phase_timings if not p.get("success", True)])
        score -= failed_phases * 10
        
        return max(0.0, min(100.0, score))