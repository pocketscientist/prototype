"""
Cell execution engine for PocketScientist agents.
"""

import os
import sys
import io
import traceback
import warnings
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import matplotlib
import matplotlib.pyplot as plt
import tempfile
from pathlib import Path
import base64


# Set matplotlib to non-interactive backend to prevent GUI issues
matplotlib.use('Agg')


@dataclass
class ExecutionResult:
    """Result of code cell execution."""
    success: bool
    output: str = ""
    error: str = ""
    plots: List[str] = None  # Base64 encoded plot images
    variables_created: List[str] = None
    execution_time: float = 0.0
    
    def __post_init__(self):
        if self.plots is None:
            self.plots = []
        if self.variables_created is None:
            self.variables_created = []


class CellExecutor:
    """Executes Python code cells in a persistent environment."""
    
    def __init__(self, working_dir: Optional[Path] = None):
        self.working_dir = working_dir or Path.cwd()
        self.namespace = self._initialize_namespace()
        self.execution_count = 0
        
        # Set working directory
        os.chdir(str(self.working_dir))
    
    def _initialize_namespace(self) -> Dict[str, Any]:
        """Initialize the execution namespace with common imports."""
        namespace = {
            '__name__': '__main__',
            '__builtins__': __builtins__,
        }
        
        # Execute common imports
        common_imports = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set up plotting
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100
"""
        
        try:
            exec(common_imports, namespace)
        except Exception as e:
            print(f"Warning: Could not import all common libraries: {e}")
        
        return namespace
    
    def execute_code(
        self, 
        code: str, 
        max_retries: int = 5,
        timeout: int = 30
    ) -> Tuple[ExecutionResult, str]:
        """
        Execute code with retry logic and result observation.
        
        Args:
            code: Python code to execute
            max_retries: Maximum number of retry attempts
            timeout: Execution timeout in seconds
            
        Returns:
            Tuple of (ExecutionResult, final_code_used)
        """
        
        original_code = code
        current_code = code
        
        for attempt in range(max_retries + 1):
            result = self._execute_single_attempt(current_code, timeout)
            
            if result.success:
                return result, current_code
            
            if attempt < max_retries:
                # Try to fix the error and retry
                fixed_code = self._attempt_error_fix(current_code, result.error)
                if fixed_code != current_code:
                    current_code = fixed_code
                    continue
                else:
                    # If we can't fix it, return the error
                    break
        
        # If all retries failed, return the last result
        return result, current_code
    
    def _execute_single_attempt(self, code: str, timeout: int) -> ExecutionResult:
        """Execute a single attempt of code execution."""
        
        import time
        start_time = time.time()
        self.execution_count += 1
        
        # Capture variables before execution
        vars_before = set(self.namespace.keys())
        
        # Clear any previous plots
        plt.clf()
        plt.close('all')
        
        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Execute the code
                exec(code, self.namespace)
            
            # Capture variables created
            vars_after = set(self.namespace.keys())
            new_vars = list(vars_after - vars_before)
            
            # Capture any plots
            plots = self._capture_plots()
            
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                success=True,
                output=stdout_capture.getvalue(),
                error="",
                plots=plots,
                variables_created=new_vars,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}"
            stderr_output = stderr_capture.getvalue()
            
            if stderr_output:
                error_msg += f"\n{stderr_output}"
            
            return ExecutionResult(
                success=False,
                output=stdout_capture.getvalue(),
                error=error_msg,
                execution_time=execution_time
            )
    
    def _capture_plots(self) -> List[str]:
        """Capture any matplotlib plots as base64 encoded images."""
        plots = []
        
        # Get all current figure numbers
        fig_nums = plt.get_fignums()
        
        for fig_num in fig_nums:
            try:
                fig = plt.figure(fig_num)
                
                # Save plot to bytes
                with tempfile.NamedTemporaryFile(suffix='.png', delete=True) as tmp:
                    fig.savefig(tmp.name, format='png', bbox_inches='tight', dpi=100)
                    
                    # Read and encode as base64
                    with open(tmp.name, 'rb') as f:
                        img_data = f.read()
                        b64_img = base64.b64encode(img_data).decode('utf-8')
                        plots.append(b64_img)
                
                plt.close(fig)
                
            except Exception as e:
                print(f"Warning: Could not capture plot {fig_num}: {e}")
        
        return plots
    
    def _attempt_error_fix(self, code: str, error: str) -> str:
        """Attempt to fix common errors in code."""
        
        fixed_code = code
        
        # Common error fixes
        if "NameError" in error and "is not defined" in error:
            # Try to add missing imports or variable definitions
            if "pd" in error:
                fixed_code = "import pandas as pd\n" + fixed_code
            elif "np" in error:
                fixed_code = "import numpy as np\n" + fixed_code
            elif "plt" in error:
                fixed_code = "import matplotlib.pyplot as plt\n" + fixed_code
        
        elif "FileNotFoundError" in error:
            # Try to use relative paths for common dataset references
            if "csv" in fixed_code:
                # Replace absolute paths with relative paths
                import re
                # Find CSV file references and make them relative
                csv_pattern = r'[\'"]([^\'"]*/)?([^/\'"]+\.csv)[\'"]'
                def make_relative(match):
                    filename = match.group(2)
                    return f'"{filename}"'
                fixed_code = re.sub(csv_pattern, make_relative, fixed_code)
        
        elif "AttributeError" in error:
            # Try to fix common attribute errors
            if "DataFrame" in error and "has no attribute" in error:
                # Might be working with wrong data type
                pass
        
        elif "KeyError" in error:
            # Column name issues - try to be more defensive
            if "columns" in fixed_code:
                # Add error checking for column existence
                fixed_code = f"""
# Check available columns first
if 'df' in locals():
    print("Available columns:", list(df.columns))
{fixed_code}
"""
        
        elif "IndentationError" in error or "SyntaxError" in error:
            # Try to fix indentation issues
            lines = fixed_code.split('\n')
            # Remove excessive indentation
            fixed_lines = []
            for line in lines:
                if line.strip():
                    # Remove leading whitespace and add minimal indent if needed
                    stripped = line.lstrip()
                    if any(line.strip().startswith(kw) for kw in ['if', 'for', 'while', 'def', 'class', 'try', 'except', 'with']):
                        fixed_lines.append(stripped + ':' if not stripped.endswith(':') else stripped)
                    else:
                        fixed_lines.append(stripped)
                else:
                    fixed_lines.append('')
            fixed_code = '\n'.join(fixed_lines)
        
        return fixed_code
    
    def get_variable_info(self, var_name: str) -> Dict[str, Any]:
        """Get information about a variable in the namespace."""
        
        if var_name not in self.namespace:
            return {"exists": False}
        
        var = self.namespace[var_name]
        info = {
            "exists": True,
            "type": type(var).__name__,
            "value": str(var)[:200] + "..." if len(str(var)) > 200 else str(var)
        }
        
        # Add specific info for pandas DataFrames
        if hasattr(var, 'shape') and hasattr(var, 'columns'):
            info.update({
                "shape": var.shape,
                "columns": list(var.columns) if hasattr(var, 'columns') else None,
                "dtypes": str(var.dtypes) if hasattr(var, 'dtypes') else None
            })
        
        return info
    
    def get_namespace_summary(self) -> Dict[str, str]:
        """Get a summary of variables in the current namespace."""
        
        summary = {}
        
        # Skip built-in and imported modules
        skip_prefixes = ('__', '_', 'pd', 'np', 'plt', 'sns', 'go', 'px')
        
        for name, obj in self.namespace.items():
            if not name.startswith(skip_prefixes) and not callable(obj):
                try:
                    obj_type = type(obj).__name__
                    if hasattr(obj, 'shape'):
                        summary[name] = f"{obj_type} {obj.shape}"
                    else:
                        obj_str = str(obj)
                        if len(obj_str) > 50:
                            obj_str = obj_str[:50] + "..."
                        summary[name] = f"{obj_type}: {obj_str}"
                except:
                    summary[name] = f"{type(obj).__name__}"
        
        return summary
    
    def clear_namespace(self, keep_imports: bool = True):
        """Clear the execution namespace, optionally keeping imports."""
        
        if keep_imports:
            # Keep only imports and built-ins
            imports_and_builtins = {}
            
            for name, obj in self.namespace.items():
                if (name.startswith('__') or 
                    callable(obj) or 
                    hasattr(obj, '__module__') or
                    name in ['pd', 'np', 'plt', 'sns', 'go', 'px']):
                    imports_and_builtins[name] = obj
            
            self.namespace = imports_and_builtins
        else:
            self.namespace = self._initialize_namespace()
        
        self.execution_count = 0