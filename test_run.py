#!/usr/bin/env python3
"""
Test script to verify PocketScientist installation and basic functionality.
"""

import sys
from pathlib import Path
import tempfile

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from pocketscientist.llm import create_llm_provider
        print("✅ LLM module imported")
    except ImportError as e:
        print(f"❌ LLM module import failed: {e}")
        return False
    
    try:
        from pocketscientist.agents import BusinessUnderstandingAgent
        print("✅ Agents module imported")
    except ImportError as e:
        print(f"❌ Agents module import failed: {e}")
        return False
    
    try:
        from pocketscientist.orchestrator import DataScienceOrchestrator
        print("✅ Orchestrator module imported")
    except ImportError as e:
        print(f"❌ Orchestrator module import failed: {e}")
        return False
    
    try:
        from pocketscientist.notebook_builder import NotebookBuilder
        print("✅ Notebook builder imported")
    except ImportError as e:
        print(f"❌ Notebook builder import failed: {e}")
        return False
    
    try:
        from pocketscientist.report_generator import ReportGenerator
        print("✅ Report generator imported")
    except ImportError as e:
        print(f"❌ Report generator import failed: {e}")
        return False
    
    try:
        from pocketscientist.utils import validate_dataset
        print("✅ Utils module imported")
    except ImportError as e:
        print(f"❌ Utils module import failed: {e}")
        return False
    
    return True

def test_dataset_validation():
    """Test dataset validation."""
    print("\nTesting dataset validation...")
    
    from pocketscientist.utils import validate_dataset
    
    # Test with sample dataset
    sample_path = project_root / "sample_data.csv"
    validation = validate_dataset(str(sample_path))
    
    if validation["valid"]:
        print("✅ Sample dataset validation passed")
        print(f"   Shape: {validation['shape']}")
        print(f"   Columns: {len(validation['columns'])}")
        return True
    else:
        print(f"❌ Sample dataset validation failed: {validation['errors']}")
        return False

def test_notebook_builder():
    """Test notebook building functionality."""
    print("\nTesting notebook builder...")
    
    from pocketscientist.notebook_builder import NotebookBuilder
    
    try:
        builder = NotebookBuilder()
        
        # Create some sample cells
        sample_cells = [
            {
                "cell_type": "markdown",
                "source": ["# Test Notebook", "", "This is a test."],
                "metadata": {}
            },
            {
                "cell_type": "code", 
                "source": ["import pandas as pd", "print('Hello, PocketScientist!')"],
                "metadata": {}
            }
        ]
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix='.ipynb', delete=False) as tmp:
            output_path = Path(tmp.name)
        
        builder.build_notebook(
            cells=sample_cells,
            output_path=output_path,
            title="Test Notebook"
        )
        
        if output_path.exists():
            print("✅ Notebook building test passed")
            output_path.unlink()  # Clean up
            return True
        else:
            print("❌ Notebook building test failed - no output file")
            return False
            
    except Exception as e:
        print(f"❌ Notebook building test failed: {e}")
        return False

def test_report_generator():
    """Test HTML report generation."""
    print("\nTesting report generator...")
    
    from pocketscientist.report_generator import ReportGenerator
    from pocketscientist.agents.base import AgentState
    
    try:
        generator = ReportGenerator()
        
        # Create mock state
        state = AgentState(
            dataset_path="test.csv",
            user_context="Test analysis"
        )
        state.completed_phases.add("business_understanding")
        state.findings["business_understanding"] = {"test": "finding"}
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp:
            output_path = Path(tmp.name)
        
        generator.generate_report(
            state=state,
            output_path=output_path
        )
        
        if output_path.exists():
            print("✅ Report generation test passed")
            output_path.unlink()  # Clean up
            return True
        else:
            print("❌ Report generation test failed - no output file")
            return False
            
    except Exception as e:
        print(f"❌ Report generation test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("PocketScientist System Test")
    print("=" * 40)
    
    tests = [
        ("Imports", test_imports),
        ("Dataset Validation", test_dataset_validation),
        ("Notebook Builder", test_notebook_builder),
        ("Report Generator", test_report_generator)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print("\n" + "=" * 40)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! PocketScientist is ready to use.")
        print("\nTo run with Ollama (requires Ollama to be running):")
        print("python -m pocketscientist.cli sample_data.csv 'what insights can you find?'")
    else:
        print("⚠️ Some tests failed. Check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)