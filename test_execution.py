#!/usr/bin/env python3
"""
Test script for cell execution functionality.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_cell_execution():
    """Test the cell execution system."""
    
    from pocketscientist.execution import CellExecutor
    
    print("Testing cell execution...")
    
    # Create executor
    executor = CellExecutor(working_dir=project_root)
    
    # Test basic execution
    code1 = """
import pandas as pd
import numpy as np
df = pd.read_csv('sample_data.csv')
print("Dataset loaded successfully!")
print("Shape:", df.shape)
print("Columns:", list(df.columns))
"""
    
    result1, final_code1 = executor.execute_code(code1, max_retries=3)
    
    print(f"\n=== Test 1: Basic Data Loading ===")
    print(f"Success: {result1.success}")
    print(f"Output: {result1.output}")
    if result1.error:
        print(f"Error: {result1.error}")
    print(f"Variables created: {result1.variables_created}")
    print(f"Execution time: {result1.execution_time:.2f}s")
    
    # Test visualization
    code2 = """
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
df['age'].hist(bins=20, alpha=0.7)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
"""
    
    result2, final_code2 = executor.execute_code(code2, max_retries=3)
    
    print(f"\n=== Test 2: Visualization ===")
    print(f"Success: {result2.success}")
    print(f"Output: {result2.output}")
    if result2.error:
        print(f"Error: {result2.error}")
    print(f"Plots generated: {len(result2.plots)}")
    print(f"Execution time: {result2.execution_time:.2f}s")
    
    # Test error handling and retry
    code3 = """
# This should initially fail but be fixed by retry logic
print("Available columns:", df.coluns)  # Typo: coluns instead of columns
"""
    
    result3, final_code3 = executor.execute_code(code3, max_retries=3)
    
    print(f"\n=== Test 3: Error Handling ===")
    print(f"Success: {result3.success}")
    print(f"Output: {result3.output}")
    if result3.error:
        print(f"Error: {result3.error}")
    print(f"Final code used: {repr(final_code3)}")
    
    # Show namespace summary
    print(f"\n=== Namespace Summary ===")
    namespace_summary = executor.get_namespace_summary()
    for var, info in namespace_summary.items():
        print(f"{var}: {info}")
    
    return result1.success and result2.success

if __name__ == "__main__":
    success = test_cell_execution()
    if success:
        print("\n🎉 Cell execution test passed!")
    else:
        print("\n❌ Cell execution test failed!")
    sys.exit(0 if success else 1)